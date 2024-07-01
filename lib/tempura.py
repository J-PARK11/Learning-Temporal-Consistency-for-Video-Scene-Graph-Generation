"""
Let's get the relationships yo
"""

import math
from re import U
import numpy as np
import torch
import json
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import math
import pickle

from tools.utils.word_vectors import obj_edge_vectors
from tools.utils.transformer import transformer
from tools.utils.fpn.box_utils import center_size
from tools.utils.gmm_heads import *
from tools.fasterRCNN.lib.model.roi_layers import ROIAlign, nms
from tools.utils.draw_rectangles.draw_rectangles import draw_union_boxes

EncoderLayer = nn.TransformerEncoderLayer
Encoder = nn.TransformerEncoder

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor, indices=None) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        if indices is None:
            x = x + self.pe[:,:x.size(1)]
        else:
            pos = torch.cat([self.pe[:, index] for index in indices])            
            x = x + pos
        return self.dropout(x)

class ObjectClassifier(nn.Module):
    """
    Module for computing the object contexts and edge contexts
    """

    def __init__(self, mode='sgdet', obj_head='gmm', K=4, obj_classes=None, 
                mem_compute=None, selection=None, selection_lambda=0.5,
                tracking=None):
        super(ObjectClassifier, self).__init__()
        self.classes = obj_classes
        self.mode = mode
        self.GMM_K =K
        self.obj_memory = []
        self.mem_compute = mem_compute
        self.selection = selection
        #----------add nms when sgdet
        self.nms_filter_duplicates = True
        self.max_per_img =64
        self.thresh = 0.01

        #roi align
        self.RCNN_roi_align = ROIAlign((7, 7), 1.0/16.0, 0)

        embed_vecs = obj_edge_vectors(obj_classes[1:], wv_type='glove.6B', wv_dir='tools/utils/word_vectors', wv_dim=200)
        self.obj_embed = nn.Embedding(len(obj_classes)-1, 200)
        self.obj_embed.weight.data = embed_vecs.clone()

        # This probably doesn't help it much
        self.pos_embed = nn.Sequential(nn.BatchNorm1d(4, momentum=0.01 / 10.0),
                                       nn.Linear(4, 128),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout(0.1))
        self.obj_dim = 2048
        self.obj_head = obj_head
        self.tracking = tracking
        mem_embed = 1024
        
        if self.tracking:
            d_model = self.obj_dim + 200 + 128
            encoder_layer = EncoderLayer(d_model=d_model, dim_feedforward=1024, nhead=8, batch_first=True)
            self.positional_encoder = PositionalEncoding(d_model, 0.1, 600 if mode=="sgdet" else 400)
            self.encoder_tran = Encoder(encoder_layer, num_layers=3)
            mem_embed = d_model
            
        if mem_compute:
                self.mem_attention = nn.MultiheadAttention(mem_embed, 1, 0.0, bias=False)
               
                if selection == 'manual':
                    self.selector = selection_lambda
                else:
                    self.selector = nn.Linear(1024,1)
        if obj_head == 'gmm':
            self.intermediate =  nn.Sequential(nn.Linear(self.obj_dim + 200 + 128, 1024),
                                               nn.BatchNorm1d(1024),
                                               nn.ReLU())
            self.decoder_lin = GMM_head(hid_dim=1024, num_classes=len(self.classes), rel_type=None, k=self.GMM_K)

        else:
            self.intermediate =  nn.Sequential(nn.Linear(self.obj_dim + 200 + 128, 1024),
                                               nn.BatchNorm1d(1024),
                                               nn.ReLU())
            self.decoder_lin = nn.Sequential(nn.Linear(1024, len(self.classes)))

    def clean_class(self, entry, b, class_idx):
        final_boxes = []
        final_dists = []
        final_feats = []
        final_mem_feats = []
        final_labels = []
        for i in range(b):
            scores = entry['distribution'][entry['boxes'][:, 0] == i]
            pred_boxes = entry['boxes'][entry['boxes'][:, 0] == i]
            feats = entry['features'][entry['boxes'][:, 0] == i]

            if 'object_mem_features' in entry.keys():
                mem_feats = entry['object_mem_features'][entry['boxes'][:, 0] == i]
            else:
                mem_feats = feats
            pred_labels = entry['pred_labels'][entry['boxes'][:, 0] == i]

            new_box = pred_boxes[entry['pred_labels'][entry['boxes'][:, 0] == i] == class_idx]
            new_feats = feats[entry['pred_labels'][entry['boxes'][:, 0] == i] == class_idx]
            new_mem_feats = mem_feats[entry['pred_labels'][entry['boxes'][:, 0] == i] == class_idx]
            new_scores = scores[entry['pred_labels'][entry['boxes'][:, 0] == i] == class_idx]
            new_scores[:, class_idx-1] = 0
            if new_scores.shape[0] > 0:
                new_labels = torch.argmax(new_scores, dim=1) + 1
            else:
                new_labels = torch.tensor([], dtype=torch.long).cuda(0)

            final_dists.append(scores)
            final_dists.append(new_scores)
            final_boxes.append(pred_boxes)
            final_boxes.append(new_box)
            final_feats.append(feats)
            final_feats.append(new_feats)
            final_mem_feats.append(mem_feats)
            final_mem_feats.append(new_mem_feats)
            final_labels.append(pred_labels)
            final_labels.append(new_labels)

        entry['boxes'] = torch.cat(final_boxes, dim=0)
        entry['distribution'] = torch.cat(final_dists, dim=0)
        entry['features'] = torch.cat(final_feats, dim=0)
        entry['object_mem_features'] = torch.cat(final_mem_feats, dim=0)
        entry['pred_labels'] = torch.cat(final_labels, dim=0)
        
        return entry

    def mem_selection(self,feat):
        if self.selection == 'manual':
            return self.selector
        else:
            return self.selector(feat).sigmoid()

    def memory_hallucinator(self,memory,feat):
        if len(memory) != 0:
            e = self.mem_selection(feat)
            q = feat.unsqueeze(1)
            
            k = v = memory.unsqueeze(1)
            mem_features,_ = self.mem_attention(q,k,v)

            if e is not None:
                mem_encoded_features = e*feat + (1-e)*mem_features.squeeze(1)
            else:
                mem_encoded_features = feat + mem_features.squeeze(1)
            # mem_encoded_features = feat + e*mem_features.squeeze(1)
        else:
            mem_encoded_features = feat

        return mem_encoded_features
    
    # OSPU
    def classify(self,entry,obj_features,phase='train',unc=False):
        if self.tracking:
            indices = entry["indices"]

            # save memory by filetering out single-element sequences, indices[0]
            final_features = torch.zeros_like(obj_features).to(obj_features.device)
            if len(indices)>1:
                pos_index = []
                for index in indices[1:]:
                    im_idx, counts = torch.unique(entry["boxes"][index][:,0].view(-1), return_counts=True, sorted=True)
                    counts = counts.tolist()
                    pos = torch.cat([torch.LongTensor([im]*count) for im, count in zip(range(len(counts)), counts)])
                    pos_index.append(pos)
                sequence_features = pad_sequence([obj_features[index] for index in indices[1:]], batch_first=True)
                masks = (1-pad_sequence([torch.ones(len(index)) for index in indices[1:]], batch_first=True)).bool()
                pos_index = pad_sequence(pos_index, batch_first=True)
                obj_ = self.encoder_tran(self.positional_encoder(sequence_features, pos_index),src_key_padding_mask=masks.cuda())
                obj_flat = torch.cat([obj[:len(index)]for index, obj in zip(indices[1:],obj_)])
                indices_flat = torch.cat(indices[1:]).unsqueeze(1).repeat(1,obj_features.shape[1])
                final_features.scatter_(0, indices_flat, obj_flat)
                
            if len(indices[0]) > 0:
                non_ = self.encoder_tran(self.positional_encoder(obj_features[indices[0]].unsqueeze(1)))           
                final_features.scatter_(0, indices[0].unsqueeze(1).repeat(1,obj_features.shape[1]), non_[:,0,:])

            obj_features = final_features
            
            entry['object_features'] = obj_features
            if self.mem_compute:
                obj_features = self.memory_hallucinator(memory=self.obj_memory, feat=obj_features)
            entry['object_mem_features'] = obj_features
            obj_features = self.intermediate(obj_features)
            
        else:
            obj_features = self.intermediate(obj_features)
            entry['object_features'] = obj_features
            if self.mem_compute:
                obj_features = self.memory_hallucinator(memory=self.obj_memory, feat=obj_features)
            entry['object_mem_features'] = obj_features

        if phase == 'train':           
            if self.obj_head == 'gmm':
                if not unc:
                    entry['distribution'] = self.decoder_lin(obj_features,phase=phase,unc=unc)
                else:
                    entry['distribution'] = self.decoder_lin(obj_features,phase='test',unc=False)
                    entry['obj_al_uc'],entry['obj_ep_uc'] = self.decoder_lin(obj_features,unc=unc)
            else:
                entry['distribution'] = self.decoder_lin(obj_features)
            entry['pred_labels'] = entry['labels']
        else:
            if self.obj_head == 'gmm':
                entry['distribution'] = self.decoder_lin(obj_features,phase=phase,unc=unc)
            else:
                entry['distribution'] = self.decoder_lin(obj_features)
                entry['distribution'] = torch.softmax(entry['distribution'][:, 1:],dim=1)
        return entry

    def forward(self, entry, phase='train', unc=False):

        if self.mode  == 'predcls':
            entry['pred_labels'] = entry['labels']
            return entry
        
        elif self.mode == 'sgcls':
            obj_embed = entry['distribution'] @ self.obj_embed.weight
            pos_embed = self.pos_embed(center_size(entry['boxes'][:, 1:]))
            obj_features = torch.cat((entry['features'], obj_embed, pos_embed), 1)
            
            if phase == 'train':
                entry = self.classify(entry,obj_features,phase,unc)
            else:
                entry = self.classify(entry,obj_features,phase,unc)

                box_idx = entry['boxes'][:,0].long()
                
                b = int(box_idx[-1] + 1)

                entry['pred_scores'], entry['pred_labels'] = torch.max(entry['distribution'][:, 1:], dim=1)
                entry['pred_labels'] = entry['pred_labels'] + 2

                # use the infered object labels for new pair idx
                HUMAN_IDX = torch.zeros([b, 1], dtype=torch.int64).to(obj_features.device)
                global_idx = torch.arange(0, entry['boxes'].shape[0])

                for i in range(b):
                    local_human_idx = torch.argmax(entry['distribution'][box_idx == i, 0]) # the local bbox index with highest human score in this frame
                    HUMAN_IDX[i] = global_idx[box_idx == i][local_human_idx]

                entry['pred_labels'][HUMAN_IDX.squeeze()] = 1
                entry['pred_scores'][HUMAN_IDX.squeeze()] = entry['distribution'][HUMAN_IDX.squeeze(), 0]

                # drop repeat overlap TODO!!!!!!!!!!!!
                for i in range(b):
                    duplicate_class = torch.mode(entry['pred_labels'][entry['boxes'][:, 0] == i])[0]
                    present = entry['boxes'][:, 0] == i
                    if torch.sum(entry['pred_labels'][entry['boxes'][:, 0] == i] ==duplicate_class) > 0:
                        duplicate_position = entry['pred_labels'][present] == duplicate_class

                        ppp = torch.argsort(entry['distribution'][present][duplicate_position][:,duplicate_class - 1])[:-1]
                        for j in ppp:

                            changed_idx = global_idx[present][duplicate_position][j]
                            entry['distribution'][changed_idx, duplicate_class-1] = 0
                            entry['pred_labels'][changed_idx] = torch.argmax(entry['distribution'][changed_idx])+1
                            entry['pred_scores'][changed_idx] = torch.max(entry['distribution'][changed_idx])


                im_idx = []  # which frame are the relations belong to
                pair = []
                for j, i in enumerate(HUMAN_IDX):
                    for m in global_idx[box_idx==j][entry['pred_labels'][box_idx==j] != 1]: # this long term contains the objects in the frame
                        im_idx.append(j)
                        pair.append([int(i), int(m)])

                pair = torch.tensor(pair).to(obj_features.device)
                im_idx = torch.tensor(im_idx, dtype=torch.float).to(obj_features.device)
                entry['pair_idx'] = pair
                entry['im_idx'] = im_idx

                entry['boxes'][:, 1:] = entry['boxes'][:, 1:] * entry['im_info']
                union_boxes = torch.cat((im_idx[:, None], torch.min(entry['boxes'][:, 1:3][pair[:, 0]], entry['boxes'][:, 1:3][pair[:, 1]]),
                                        torch.max(entry['boxes'][:, 3:5][pair[:, 0]], entry['boxes'][:, 3:5][pair[:, 1]])), 1)

                union_feat = self.RCNN_roi_align(entry['fmaps'], union_boxes)
                entry['boxes'][:, 1:] = entry['boxes'][:, 1:] / entry['im_info']
                pair_rois = torch.cat((entry['boxes'][pair[:, 0], 1:], entry['boxes'][pair[:, 1], 1:]),
                                      1).data.cpu().numpy()
                spatial_masks = torch.tensor(draw_union_boxes(pair_rois, 27) - 0.5).to(obj_features.device)
                entry['union_feat'] = union_feat
                entry['union_box'] = union_boxes
                entry['spatial_masks'] = spatial_masks
            return entry
        
        else:
            obj_embed = entry['distribution'] @ self.obj_embed.weight
            pos_embed = self.pos_embed(center_size(entry['boxes'][:, 1:]))
            obj_features = torch.cat((entry['features'], obj_embed, pos_embed), 1)
            if phase == 'train':
                entry = self.classify(entry,obj_features,phase,unc)
            else:
                entry = self.classify(entry,obj_features,phase,unc)

                box_idx = entry['boxes'][:, 0].long()
                b = int(box_idx[-1] + 1)

                entry = self.clean_class(entry, b, 5)
                entry = self.clean_class(entry, b, 8)
                entry = self.clean_class(entry, b, 17)

                # # NMS
                final_boxes = []
                final_dists = []
                final_feats = []
                final_mem_feats = []
                for i in range(b):
                    # images in the batch
                    scores = entry['distribution'][entry['boxes'][:, 0] == i]
                    pred_boxes = entry['boxes'][entry['boxes'][:, 0] == i, 1:]
                    feats = entry['features'][entry['boxes'][:, 0] == i]
                    if 'object_mem_features' in entry.keys():
                        mem_feats = entry['object_mem_features'][entry['boxes'][:, 0] == i]
                    else:
                        mem_feats = entry['features'][entry['boxes'][:, 0] == i]
                    for j in range(len(self.classes) - 1):
                        # print('scores_shape: ',scores.shape,' for class ',j)
                        
                        # NMS according to obj categories
                        if scores.numel() == 0:
                            inds = torch.empty(scores.shape)
                        else:
                            inds = torch.nonzero(torch.argmax(scores, dim=1) == j).view(-1)
                        
                        # if there is det
                        if inds.numel() > 0:
                            cls_dists = scores[inds]
                            cls_feats = feats[inds]
                            cls_mem_feats = mem_feats[inds]
                            cls_scores = cls_dists[:, j]
                            _, order = torch.sort(cls_scores, 0, True)
                            cls_boxes = pred_boxes[inds]
                            cls_dists = cls_dists[order]
                            cls_feats = cls_feats[order]
                            cls_mem_feats = cls_mem_feats[order]
                            keep = nms(cls_boxes[order, :], cls_scores[order], 0.6)  # hyperparameter

                            final_dists.append(cls_dists[keep.view(-1).long()])
                            final_boxes.append(torch.cat((torch.tensor([[i]],dtype=torch.float).repeat(keep.shape[0],1).cuda(0), cls_boxes[order, :][keep.view(-1).long()]), 1))
                            final_feats.append(cls_feats[keep.view(-1).long()])
                            final_mem_feats.append(cls_mem_feats[keep.view(-1).long()])

                entry['boxes'] = torch.cat(final_boxes, dim=0)
                box_idx = entry['boxes'][:, 0].long()
                entry['distribution'] = torch.cat(final_dists, dim=0)
                entry['features'] = torch.cat(final_feats, dim=0)
                entry['object_mem_features'] = torch.cat(final_mem_feats, dim=0)

                entry['pred_scores'], entry['pred_labels'] = torch.max(entry['distribution'][:, 1:], dim=1)
                entry['pred_labels'] = entry['pred_labels'] + 2

                # use the infered object labels for new pair idx
                HUMAN_IDX = torch.zeros([b, 1], dtype=torch.int64).to(box_idx.device)
                global_idx = torch.arange(0, entry['boxes'].shape[0])

                for i in range(b):
                    if entry['distribution'][ box_idx == i, 0].numel() > 0:
                        local_human_idx = torch.argmax(entry['distribution'][box_idx == i, 0])  # the local bbox index with highest human score in this frame
                        HUMAN_IDX[i] = global_idx[box_idx == i][local_human_idx]

                entry['pred_labels'][HUMAN_IDX.squeeze()] = 1
                entry['pred_scores'][HUMAN_IDX.squeeze()] = entry['distribution'][HUMAN_IDX.squeeze(), 0]

                im_idx = []  # which frame are the relations belong to
                pair = []
                for j, i in enumerate(HUMAN_IDX):
                    for m in global_idx[box_idx == j][
                        entry['pred_labels'][box_idx == j] != 1]:  # this long term contains the objects in the frame
                        im_idx.append(j)
                        pair.append([int(i), int(m)])

                pair = torch.tensor(pair).to(box_idx.device)
                im_idx = torch.tensor(im_idx, dtype=torch.float).to(box_idx.device)
                entry['pair_idx'] = pair
                entry['im_idx'] = im_idx
                entry['human_idx'] = HUMAN_IDX
                entry['boxes'][:, 1:] = entry['boxes'][:, 1:] * entry['im_info']
                union_boxes = torch.cat(
                    (im_idx[:, None], torch.min(entry['boxes'][:, 1:3][pair[:, 0]], entry['boxes'][:, 1:3][pair[:, 1]]),
                     torch.max(entry['boxes'][:, 3:5][pair[:, 0]], entry['boxes'][:, 3:5][pair[:, 1]])), 1)

                union_feat = self.RCNN_roi_align(entry['fmaps'], union_boxes)
                entry['boxes'][:, 1:] = entry['boxes'][:, 1:] / entry['im_info']
                entry['union_feat'] = union_feat
                entry['union_box'] = union_boxes
                pair_rois = torch.cat((entry['boxes'][pair[:, 0], 1:], entry['boxes'][pair[:, 1], 1:]),
                                      1).data.cpu().numpy()
                entry['spatial_masks'] = torch.tensor(draw_union_boxes(pair_rois, 27) - 0.5).to(box_idx.device)
            
            return entry


class TEMPURA(nn.Module):

    def __init__(self, mode='sgdet',attention_class_num=None, spatial_class_num=None, \
                 contact_class_num=None, obj_classes=None,
                 rel_classes=None,enc_layer_num=None, dec_layer_num=None, obj_mem_compute=None,rel_mem_compute=None,
                 mem_fusion=None,selection=None,selection_lambda=0.5,take_obj_mem_feat=False,
                 obj_head = 'gmm', rel_head = 'gmm',K =None, tracking=None):

        """
        :param classes: Object classes
        :param rel_classes: Relationship classes. None if were not using rel mode
        :param mode: (sgcls, predcls, or sgdet)
        """
        super(TEMPURA, self).__init__()
        self.obj_classes = obj_classes
        self.GMM_K = K
        self.mem_fusion = mem_fusion
        self.rel_classes = rel_classes
        self.attention_class_num = attention_class_num
        self.spatial_class_num = spatial_class_num
        self.contact_class_num = contact_class_num
        
        assert mode in ('sgdet', 'sgcls', 'predcls')
        self.mode = mode
        self.tracking = tracking
        self.take_obj_mem_feat = take_obj_mem_feat

        self.obj_head = obj_head
        self.rel_head = rel_head
        self.obj_mem_compute = obj_mem_compute
        self.rel_mem_compute = rel_mem_compute

        self.selection_lambda = selection_lambda
        self.rel_memory = []
        self.object_classifier = ObjectClassifier(mode=self.mode, obj_classes=self.obj_classes, 
                                                obj_head = obj_head, mem_compute=obj_mem_compute, K=K, selection=selection,
                                                selection_lambda=selection_lambda, tracking=self.tracking)

        ###################################
        self.union_func1 = nn.Conv2d(1024, 256, 1, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(2, 256 //2, kernel_size=7, stride=2, padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256//2, momentum=0.01),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(256 // 2, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256, momentum=0.01),
        )
        if not take_obj_mem_feat:
            self.subj_fc = nn.Linear(2048, 512)
            self.obj_fc = nn.Linear(2048, 512)
        else:
            if self.tracking:
                self.subj_fc = nn.Linear(2048+200+128, 512)
                self.obj_fc = nn.Linear(2048+200+128, 512)
            else:
                self.subj_fc = nn.Linear(1024, 512)
                self.obj_fc = nn.Linear(1024, 512)

        self.vr_fc = nn.Linear(256*7*7, 512)

        embed_vecs = obj_edge_vectors(obj_classes, wv_type='glove.6B', wv_dir='tools/utils/word_vectors', wv_dim=200)
        self.obj_embed = nn.Embedding(len(obj_classes), 200)
        self.obj_embed.weight.data = embed_vecs.clone()

        self.obj_embed2 = nn.Embedding(len(obj_classes), 200)
        self.obj_embed2.weight.data = embed_vecs.clone()

            
        self.glocal_transformer = transformer(enc_layer_num=enc_layer_num, dec_layer_num=dec_layer_num,
                                              embed_dim=1936, nhead=8,
                                              dim_feedforward=2048, dropout=0.1, mode='latter', 
                                              mem_compute=rel_mem_compute, mem_fusion=mem_fusion,
                                              selection=selection, selection_lambda=self.selection_lambda)

        if rel_head == 'gmm':
            self.a_rel_compress = GMM_head(hid_dim=1936, num_classes=self.attention_class_num, rel_type='attention', k=self.GMM_K)
            self.s_rel_compress = GMM_head(hid_dim=1936, num_classes=self.spatial_class_num, rel_type='spatial', k=self.GMM_K)
            self.c_rel_compress = GMM_head(hid_dim=1936, num_classes=self.contact_class_num, rel_type='contact', k=self.GMM_K)

        else:
            self.a_rel_compress = nn.Linear(1936, self.attention_class_num)
            self.s_rel_compress = nn.Linear(1936, self.spatial_class_num)
            self.c_rel_compress = nn.Linear(1936, self.contact_class_num)

    def forward(self, entry, phase='train',unc=False):

        # OSPU
        entry = self.object_classifier(entry, phase=phase, unc=unc)
        """
        < Independent Object >
        features: [Object, 2048]
        pred_labels: [Object]
        
        
        < Union Pair >
        union_feat: [Pair, 1024, 7, 7]
        union_box: [Pair, 5]
        Spaital_masks: [Pair, 2, 27, 27]
        """
        
        """ : Visualize Spatial Masks
        import cv2
        for i in range(entry['spatial_masks'].shape[0]):
            for j in range(2):
                cv2.imwrite('output{}_{}.png'.format(i,j), np.array(entry['spatial_masks'][i,j].cpu().detach())*255.0)
        """
        
        # Visual part
        if not self.take_obj_mem_feat:
            subj_rep = entry['features'][entry['pair_idx'][:, 0]]       # [71, 2048]: Subject Backbone Features
            obj_rep = entry['features'][entry['pair_idx'][:, 1]]        # [71, 2048]: Object Backbone Features
        else:
            subj_rep = entry['object_mem_features'][entry['pair_idx'][:, 0]]
            obj_rep = entry['object_mem_features'][entry['pair_idx'][:, 1]]

        subj_rep = self.subj_fc(subj_rep)                               # [71, 512]
        obj_rep = self.obj_fc(obj_rep)                                  # [71, 512]
        
        # vr: Pair Union Feature와 Spatial Mask에 대해 각각 Conv Layer를 투영.
        # vr = self.union_func1(entry['union_feat']) + self.conv(entry['spatial_masks'])  # [71, 256, 7, 7]
        vr = self.union_func1(entry['union_feat']) + self.conv(entry['spatial_masks'])  # [71, 256, 7, 7]
        vr = self.vr_fc(vr.view(-1,256*7*7))                                            # [71, 512]
        
        x_visual = torch.cat((subj_rep, obj_rep, vr), 1)                # [71, 1536]

        # Semantic part
        subj_class = entry['pred_labels'][entry['pair_idx'][:, 0]]      # [71]   
        obj_class = entry['pred_labels'][entry['pair_idx'][:, 1]]       # [71]

        # Word Embedding
        subj_emb = self.obj_embed(subj_class)                           # [71, 200]
        obj_emb = self.obj_embed2(obj_class)                            # [71, 200]
        x_semantic = torch.cat((subj_emb, obj_emb), 1)                  # [71, 400]

        # Visual & Semantic Features
        rel_features = torch.cat((x_visual, x_semantic), dim=1)         # [71, 1936]

        # Spatial-Temporal Transformer
        global_output, rel_features, mem_features, _, _ = self.glocal_transformer(features=rel_features, im_idx=entry['im_idx'], memory=self.rel_memory)
        """
        PEG, MDU Output Config
            => global_output: [71, 1936]
            => rel_features: [71, 1936]    
            => mem_features: [71, 1936]
        """
        
        # PEG, MDU Output
        entry['obj_class'] = obj_class
        entry["rel_features"] = rel_features
        entry['rel_mem_features'] = mem_features

        # Predicate Label이 형성되는 부분.
        if self.rel_head == 'gmm':
            if not unc:
                entry["attention_distribution"] = self.a_rel_compress(global_output,phase,unc)  # [71, 3]
                entry["spatial_distribution"] = self.s_rel_compress(global_output,phase,unc)    # [71, 6]
                entry["contacting_distribution"] = self.c_rel_compress(global_output,phase,unc) # [71, 17]
            else:
                entry["attention_al_uc"], entry["attention_ep_uc"] = self.a_rel_compress(global_output,phase,unc)
                entry["spatial_al_uc"], entry["spatial_ep_uc"] = self.s_rel_compress(global_output,phase,unc)
                entry["contacting_al_uc"], entry["contacting_ep_uc"] = self.c_rel_compress(global_output,phase,unc)
        else:
            entry["attention_distribution"] = self.a_rel_compress(global_output)
            if phase == 'test':
                entry["attention_distribution"] = entry["attention_distribution"].softmax(-1)
            entry["spatial_distribution"] = self.s_rel_compress(global_output)
            entry["contacting_distribution"] = self.c_rel_compress(global_output)
            entry["spatial_distribution"] = torch.sigmoid(entry["spatial_distribution"])
            entry["contacting_distribution"] = torch.sigmoid(entry["contacting_distribution"])
            
        return entry
