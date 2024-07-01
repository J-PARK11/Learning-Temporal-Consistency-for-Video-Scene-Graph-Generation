import os
import cv2
import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from tools.utils.funcs import assign_relations
from tools.fasterRCNN.lib.model.roi_layers import nms
from tools.fasterRCNN.lib.model.faster_rcnn.resnet import resnet
from tools.utils.draw_rectangles.draw_rectangles import draw_union_boxes
from tools.fasterRCNN.lib.model.rpn.bbox_transform import bbox_transform_inv, clip_boxes

class detector(nn.Module):

    '''first part: object detection (image/video)'''

    def __init__(self, train, object_classes, use_SUPPLY, mode='predcls'):
        super(detector, self).__init__()

        self.is_train = train
        self.use_SUPPLY = use_SUPPLY
        self.object_classes = object_classes
        self.mode = mode

        # FasteRCNN ResNet101 Backbone Model
        self.fasterRCNN = resnet(classes=self.object_classes, num_layers=101, pretrained=False, class_agnostic=False)
        self.fasterRCNN.create_architecture()
        checkpoint = torch.load('tools/fasterRCNN/models/faster_rcnn_ag.pth')
        self.fasterRCNN.load_state_dict(checkpoint['model'])

        # Copy of ROI_Align, RCNN_Head
        self.ROI_Align = copy.deepcopy(self.fasterRCNN.RCNN_roi_align)  # RoI Aligned Feature
        self.RCNN_Head = copy.deepcopy(self.fasterRCNN._head_to_tail)

    # Filter Non-reliable boxes by FINAL_SCORES
    def filter_non_reliable_boxes(self, FINAL_BBOXES, FINAL_SCORES, FINAL_DISTRIBUTIONS, PRED_LABELS, FINAL_FEATURES, thr=0.9):
        
        valid_idx = (FINAL_SCORES > thr)
        FINAL_BBOXES = FINAL_BBOXES[valid_idx]
        FINAL_SCORES = FINAL_SCORES[valid_idx]
        FINAL_DISTRIBUTIONS = FINAL_DISTRIBUTIONS[valid_idx]
        PRED_LABELS = PRED_LABELS[valid_idx]
        FINAL_FEATURES = FINAL_FEATURES[valid_idx]
        
        return FINAL_BBOXES, FINAL_SCORES, FINAL_DISTRIBUTIONS, PRED_LABELS, FINAL_FEATURES
    
    def forward(self, im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all):
        """
        im_data: [38, 3, 1067, 600] = [Frames, C, H, W]
        in_info: [38, 3] = [Frames, (H, W, Scale factor from Raw image)]
        gt_boxes: [38, 1, 5] => 비어있음. all zero
        num_boxes: [38] => 비어있음. all zero
        gt_annotation: [38, dict(person, label)]
        """
        if self.mode == 'sgdet':
            counter = 0
            counter_image = 0

            # Create saved-bbox, labels, scores, features
            FINAL_BBOXES = torch.tensor([]).cuda(0)
            FINAL_LABELS = torch.tensor([], dtype=torch.int64).cuda(0)
            FINAL_SCORES = torch.tensor([]).cuda(0)
            FINAL_FEATURES = torch.tensor([]).cuda(0)
            FINAL_BASE_FEATURES = torch.tensor([]).cuda(0)

            # Loop for All frames in Video: 한 번에 10 Frame씩 처리함.
            while counter < im_data.shape[0]:
                # Compute 10 images in batch and collect all frames data in the video
                # 10 Images = 1 Batch for FasterRCNN
                if counter + 10 < im_data.shape[0]:
                    inputs_data = im_data[counter:counter + 10]
                    inputs_info = im_info[counter:counter + 10]
                    inputs_gtboxes = gt_boxes[counter:counter + 10]
                    inputs_numboxes = num_boxes[counter:counter + 10]

                else:
                    inputs_data = im_data[counter:]
                    inputs_info = im_info[counter:]
                    inputs_gtboxes = gt_boxes[counter:]
                    inputs_numboxes = num_boxes[counter:]

                # FasterRCNN Inference
                rois, cls_prob, bbox_pred, base_feat, roi_features = self.fasterRCNN(inputs_data, inputs_info,
                                                                                     inputs_gtboxes, inputs_numboxes)
                """
                FasterRCNN Output Configs:
                    => rois: [10, 100, 5]
                    => roi_features: [10, 100, 2048]
                    => base_feat: [10, 1024, 67, 38]
                    => bbox_pred: [10, 100, 148]
                    => cls_prob: [10, 100, 37]
                """

                SCORES = cls_prob.data              # [10, 100, 37]: Object Class Predication Score in 37 classes
                boxes = rois.data[:, :, 1:5]        # [10, 100, 4]: Object Detect BBox
                
                # bbox regression (class specific)
                box_deltas = bbox_pred.data
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor([0.1, 0.1, 0.2, 0.2]).cuda(0) \
                             + torch.FloatTensor([0.0, 0.0, 0.0, 0.0]).cuda(0)  # the first is normalize std, the second is mean
                             
                box_deltas = box_deltas.view(-1, rois.shape[1], 4 * len(self.object_classes))  # post_NMS_NTOP: 30
                pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                PRED_BOXES = clip_boxes(pred_boxes, im_info.data, 1)
                PRED_BOXES /= inputs_info[0, 2] # original bbox scale!!!!!!!!!!!!!!

                # traverse frames
                for i in range(rois.shape[0]):  # 10
                    
                    # images in the batch
                    scores = SCORES[i]              # scores: [100, 37]
                    pred_boxes = PRED_BOXES[i]      # pred_boxes: [100, 148]

                    for j in range(1, len(self.object_classes)):    # 37
                        
                        # NMS according to obj categories
                        inds = torch.nonzero(scores[:, j] > 0.1).view(-1) #0.05 is score threshold / actual paremeter: 0.1
                        
                        # if there is det
                        if inds.numel() > 0:
                            
                            cls_scores = scores[:, j][inds]
                            _, order = torch.sort(cls_scores, 0, True)
                            cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
                            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                            cls_dets = cls_dets[order]
                            keep = nms(cls_boxes[order, :], cls_scores[order], 0.4) # NMS threshold
                            cls_dets = cls_dets[keep.view(-1).long()]

                            if j == 1:
                                # for person we only keep the highest score for person!
                                final_bbox = cls_dets[0,0:4].unsqueeze(0)
                                final_score = cls_dets[0,4].unsqueeze(0)
                                final_labels = torch.tensor([j]).cuda(0)
                                final_features = roi_features[i, inds[order[keep][0]]].unsqueeze(0)
                            
                            else:
                                # for Obejct without person
                                final_bbox = cls_dets[:, 0:4]
                                final_score = cls_dets[:, 4]
                                final_labels = torch.tensor([j]).repeat(keep.shape[0]).cuda(0)
                                final_features = roi_features[i, inds[order[keep]]]

                            final_bbox = torch.cat((torch.tensor([[counter_image]], dtype=torch.float).repeat(final_bbox.shape[0], 1).cuda(0), final_bbox), 1)
                            
                            FINAL_BBOXES = torch.cat((FINAL_BBOXES, final_bbox), 0)
                            FINAL_LABELS = torch.cat((FINAL_LABELS, final_labels), 0)
                            FINAL_SCORES = torch.cat((FINAL_SCORES, final_score), 0)
                            FINAL_FEATURES = torch.cat((FINAL_FEATURES, final_features), 0)

                    FINAL_BASE_FEATURES = torch.cat((FINAL_BASE_FEATURES, base_feat[i].unsqueeze(0)), 0)
                    counter_image += 1
                counter += 10
            
            # Final Pred Boxes
            FINAL_BBOXES = torch.clamp(FINAL_BBOXES, 0)
            prediction = {'FINAL_BBOXES': FINAL_BBOXES, 'FINAL_LABELS': FINAL_LABELS, 'FINAL_SCORES': FINAL_SCORES,
                          'FINAL_FEATURES': FINAL_FEATURES, 'FINAL_BASE_FEATURES': FINAL_BASE_FEATURES}
            """
            Prediction:
                => FINAL_BBOXES: [320, 5]
                => FINAL_LABELS: [320]
                => FINAL_SCORES: [320]
            """
            
            if self.is_train:

                DETECTOR_FOUND_IDX, GT_RELATIONS, SUPPLY_RELATIONS, assigned_labels = assign_relations(prediction, gt_annotation, assign_IOU_threshold=0.5)

                if self.use_SUPPLY:
                    # supply the unfounded gt boxes by detector into the scene graph generation training
                    FINAL_BBOXES_X = torch.tensor([]).cuda(0)
                    FINAL_LABELS_X = torch.tensor([], dtype=torch.int64).cuda(0)
                    FINAL_SCORES_X = torch.tensor([]).cuda(0)
                    FINAL_FEATURES_X = torch.tensor([]).cuda(0)
                    assigned_labels = torch.tensor(assigned_labels, dtype=torch.long).to(FINAL_BBOXES_X.device)

                    for i, j in enumerate(SUPPLY_RELATIONS):
                        if len(j) > 0:
                            unfound_gt_bboxes = torch.zeros([len(j), 5]).cuda(0)
                            unfound_gt_classes = torch.zeros([len(j)], dtype=torch.int64).cuda(0)
                            one_scores = torch.ones([len(j)], dtype=torch.float32).cuda(0)  # probability
                            for m, n in enumerate(j):
                                # if person box is missing or objects
                                if 'bbox' in n.keys():
                                    unfound_gt_bboxes[m, 1:] = torch.tensor(n['bbox']).cuda(0) * im_info[
                                        i, 2]  # don't forget scaling!
                                    unfound_gt_classes[m] = n['class']
                                else:
                                    # here happens always that IOU <0.5 but not unfounded
                                    unfound_gt_bboxes[m, 1:] = torch.tensor(n['person_bbox']).cuda(0) * im_info[
                                        i, 2]  # don't forget scaling!
                                    unfound_gt_classes[m] = 1  # person class index

                            DETECTOR_FOUND_IDX[i] = list(np.concatenate((DETECTOR_FOUND_IDX[i],
                                                                         np.arange(
                                                                             start=int(sum(FINAL_BBOXES[:, 0] == i)),
                                                                             stop=int(
                                                                                 sum(FINAL_BBOXES[:, 0] == i)) + len(
                                                                                 SUPPLY_RELATIONS[i]))), axis=0).astype('int64'))

                            GT_RELATIONS[i].extend(SUPPLY_RELATIONS[i])

                            # compute the features of unfound gt_boxes
                            pooled_feat = self.fasterRCNN.RCNN_roi_align(FINAL_BASE_FEATURES[i].unsqueeze(0),
                                                                         unfound_gt_bboxes.cuda(0))
                            pooled_feat = self.fasterRCNN._head_to_tail(pooled_feat)
                            cls_prob = F.softmax(self.fasterRCNN.RCNN_cls_score(pooled_feat), 1)

                            unfound_gt_bboxes[:, 0] = i
                            unfound_gt_bboxes[:, 1:] = unfound_gt_bboxes[:, 1:] / im_info[i, 2]
                            FINAL_BBOXES_X = torch.cat(
                                (FINAL_BBOXES_X, FINAL_BBOXES[FINAL_BBOXES[:, 0] == i], unfound_gt_bboxes))
                            FINAL_LABELS_X = torch.cat((FINAL_LABELS_X, assigned_labels[FINAL_BBOXES[:, 0] == i],
                                                        unfound_gt_classes))  # final label is not gt!
                            FINAL_SCORES_X = torch.cat(
                                (FINAL_SCORES_X, FINAL_SCORES[FINAL_BBOXES[:, 0] == i], one_scores))
                            FINAL_FEATURES_X = torch.cat(
                                (FINAL_FEATURES_X, FINAL_FEATURES[FINAL_BBOXES[:, 0] == i], pooled_feat))
                        else:
                            FINAL_BBOXES_X = torch.cat((FINAL_BBOXES_X, FINAL_BBOXES[FINAL_BBOXES[:, 0] == i]))
                            FINAL_LABELS_X = torch.cat((FINAL_LABELS_X, assigned_labels[FINAL_BBOXES[:, 0] == i]))
                            FINAL_SCORES_X = torch.cat((FINAL_SCORES_X, FINAL_SCORES[FINAL_BBOXES[:, 0] == i]))
                            FINAL_FEATURES_X = torch.cat((FINAL_FEATURES_X, FINAL_FEATURES[FINAL_BBOXES[:, 0] == i]))

                FINAL_DISTRIBUTIONS = torch.softmax(self.fasterRCNN.RCNN_cls_score(FINAL_FEATURES_X)[:, 1:], dim=1)
                global_idx = torch.arange(start=0, end=FINAL_BBOXES_X.shape[0])  # all bbox indices

                im_idx = []  # which frame are the relations belong to
                pair = []
                a_rel = []
                s_rel = []
                c_rel = []
                for i, j in enumerate(DETECTOR_FOUND_IDX):

                    for k, kk in enumerate(GT_RELATIONS[i]):
                        if 'person_bbox' in kk.keys():
                            kkk = k
                            break
                    localhuman = int(global_idx[FINAL_BBOXES_X[:, 0] == i][kkk])

                    for m, n in enumerate(j):
                        if 'class' in GT_RELATIONS[i][m].keys():
                            im_idx.append(i)

                            pair.append([localhuman, int(global_idx[FINAL_BBOXES_X[:, 0] == i][int(n)])])

                            a_rel.append(GT_RELATIONS[i][m]['attention_relationship'].tolist())
                            s_rel.append(GT_RELATIONS[i][m]['spatial_relationship'].tolist())
                            c_rel.append(GT_RELATIONS[i][m]['contacting_relationship'].tolist())

                pair = torch.tensor(pair).cuda(0)
                im_idx = torch.tensor(im_idx, dtype=torch.float).cuda(0)
                union_boxes = torch.cat((im_idx[:, None],
                                         torch.min(FINAL_BBOXES_X[:, 1:3][pair[:, 0]],
                                                   FINAL_BBOXES_X[:, 1:3][pair[:, 1]]),
                                         torch.max(FINAL_BBOXES_X[:, 3:5][pair[:, 0]],
                                                   FINAL_BBOXES_X[:, 3:5][pair[:, 1]])), 1)

                union_boxes[:, 1:] = union_boxes[:, 1:] * im_info[0, 2]
                union_feat = self.fasterRCNN.RCNN_roi_align(FINAL_BASE_FEATURES, union_boxes)

                pair_rois = torch.cat((FINAL_BBOXES_X[pair[:,0],1:],FINAL_BBOXES_X[pair[:,1],1:]), 1).data.cpu().numpy()
                spatial_masks = torch.tensor(draw_union_boxes(pair_rois, 27) - 0.5).to(FINAL_FEATURES.device)

                entry = {'boxes': FINAL_BBOXES_X,
                         'labels': FINAL_LABELS_X,
                         'scores': FINAL_SCORES_X,
                         'distribution': FINAL_DISTRIBUTIONS,
                         'im_idx': im_idx,
                         'pair_idx': pair,
                         'features': FINAL_FEATURES_X,
                         'union_feat': union_feat,
                         'spatial_masks': spatial_masks,
                         'attention_gt': a_rel,
                         'spatial_gt': s_rel,
                         'contacting_gt': c_rel}

                return entry

            else:
                FINAL_DISTRIBUTIONS = torch.softmax(self.fasterRCNN.RCNN_cls_score(FINAL_FEATURES)[:, 1:], dim=1)
                FINAL_SCORES, PRED_LABELS = torch.max(FINAL_DISTRIBUTIONS, dim=1)
                PRED_LABELS = PRED_LABELS + 1
                
                ####### Filter Non Reliable Boxes #######
                
                # FINAL_BBOXES, FINAL_SCORES, FINAL_DISTRIBUTIONS, PRED_LABELS, FINAL_FEATURES = self.filter_non_reliable_boxes(FINAL_BBOXES, FINAL_SCORES, FINAL_DISTRIBUTIONS, PRED_LABELS, FINAL_FEATURES, 0.9)
                
                #########################################

                entry = {'boxes': FINAL_BBOXES,                 # [320, 5]: Image index & BBox(4)           => GT: [109, 5], 0.5: [273, 5], 0.9: [160, 5]
                         'scores': FINAL_SCORES,                # [320]: Score of Object Clf
                         'distribution': FINAL_DISTRIBUTIONS,   # [320, 36]: Distribution of Object Clf
                         'pred_labels': PRED_LABELS,            # [320]: Pred label of Object
                         'features': FINAL_FEATURES,            # [320, 2048]: RoI features
                         'fmaps': FINAL_BASE_FEATURES,          # [38, 1024, 67, 38]: RoI base features
                         'im_info': im_info[0, 2]}

                return entry
        
        # Predcls & Sgcls
        else:
            # how many bboxes we have
            bbox_num = 0
            
            # which frame are the relations belong to
            im_idx, pair = [], []
            a_rel, s_rel, c_rel = [], [], []

            for i in gt_annotation:
                bbox_num += len(i)
              
            # Num of Object followed Variables
            FINAL_BBOXES = torch.zeros([bbox_num,5], dtype=torch.float32).cuda(0)
            FINAL_LABELS = torch.zeros([bbox_num], dtype=torch.int64).cuda(0)
            FINAL_SCORES = torch.ones([bbox_num], dtype=torch.float32).cuda(0)
            HUMAN_IDX = torch.zeros([len(gt_annotation),1], dtype=torch.int64).cuda(0)

            # GT BBox and Predicate Label Definition
            bbox_idx = 0
            for i, j in enumerate(gt_annotation):
                for m in j:
                    if 'person_bbox' in m.keys():
                        FINAL_BBOXES[bbox_idx,1:] = torch.from_numpy(m['person_bbox'][0])
                        FINAL_BBOXES[bbox_idx, 0] = i
                        FINAL_LABELS[bbox_idx] = 1
                        HUMAN_IDX[i] = bbox_idx
                        bbox_idx += 1
                    else:
                        FINAL_BBOXES[bbox_idx,1:] = torch.from_numpy(m['bbox'])
                        FINAL_BBOXES[bbox_idx, 0] = i
                        FINAL_LABELS[bbox_idx] = m['class']
                        im_idx.append(i)
                        pair.append([int(HUMAN_IDX[i]), bbox_idx])
                        a_rel.append(m['attention_relationship'].tolist())
                        s_rel.append(m['spatial_relationship'].tolist())
                        c_rel.append(m['contacting_relationship'].tolist())
                        bbox_idx += 1
            
            pair = torch.tensor(pair).cuda(0)
            im_idx = torch.tensor(im_idx, dtype=torch.float).cuda(0)

            counter = 0
            FINAL_BASE_FEATURES = torch.tensor([]).cuda(0)

            # Compute 10 images in batch and  collect all frames data in the video
            while counter < im_data.shape[0]:
                if counter + 10 < im_data.shape[0]:
                    inputs_data = im_data[counter:counter + 10]
                else:
                    inputs_data = im_data[counter:]
                    
                # Feed Image to extract Base Features
                base_feat = self.fasterRCNN.RCNN_base(inputs_data)
                FINAL_BASE_FEATURES = torch.cat((FINAL_BASE_FEATURES, base_feat), 0)    # (38, 1024, 67, 38)
                counter += 10

            # Scale to Original Image Size
            FINAL_BBOXES[:, 1:] = FINAL_BBOXES[:, 1:] * im_info[0, 2]   # (109, 5)
            
            # ROI Align: Scaling and Quantization
            #  => 이미지에서 추출한 Feature Map으로부터 BBox를 기준으로 해당 영역에 대한 Feature를 추출.
            FINAL_FEATURES = self.fasterRCNN.RCNN_roi_align(FINAL_BASE_FEATURES, FINAL_BBOXES)  # (109, 1024, 7, 7)
            FINAL_FEATURES = self.fasterRCNN._head_to_tail(FINAL_FEATURES)  # (109, 2048)

            if self.mode == 'predcls':
                
                # Union Boxes: 제안된 BBox를 통해 Pairwise 간 Union Box를 계산한다.
                union_boxes = torch.cat((im_idx[:, None], torch.min(FINAL_BBOXES[:, 1:3][pair[:, 0]], FINAL_BBOXES[:, 1:3][pair[:, 1]]),    # (71, 5)
                                         torch.max(FINAL_BBOXES[:, 3:5][pair[:, 0]], FINAL_BBOXES[:, 3:5][pair[:, 1]])), 1)
                
                # Union Box로 정의된 BBox를 기준으로 위에서와 마찬가지로 Feature Base Map을 활용해 RoI Align 및 피쳐화 수행
                union_feat = self.fasterRCNN.RCNN_roi_align(FINAL_BASE_FEATURES, union_boxes)               # (71, 1024, 7, 7)
                FINAL_BBOXES[:, 1:] = FINAL_BBOXES[:, 1:] / im_info[0, 2]
                
                pair_rois = torch.cat((FINAL_BBOXES[pair[:, 0], 1:], FINAL_BBOXES[pair[:, 1], 1:]), 1).data.cpu().numpy()   # (71, 8)
                spatial_masks = torch.tensor(draw_union_boxes(pair_rois, 27) - 0.5).to(FINAL_FEATURES.device)               # (71, 2, 27, 27)

                entry = {'boxes': FINAL_BBOXES,             # GT BBoxes: [109, 5]
                         'labels': FINAL_LABELS,            # GT Object label: [109]
                         'scores': FINAL_SCORES,            # GT Object clf scores: [109] = 1
                         'base_feat': FINAL_BASE_FEATURES,
                         'im_idx': im_idx,                  # Im idx of pair: [71]
                         'pair_idx': pair,                  # Pair Idx: [71, 2]
                         'human_idx': HUMAN_IDX,            # Human Idx: [38, 1]
                         'features': FINAL_FEATURES,        # RoI features: [109, 2048]
                         'union_feat': union_feat,          # Pair RoI Features: [71, 1024, 7, 7]
                         'union_box': union_boxes,          # GT Pair Union Boxes: [71, 5]
                         'spatial_masks': spatial_masks,    # GT Pair Spatial masks:[71, 2, 27, 27]
                         'attention_gt': a_rel,             # Attention Predicate GT: [71, M]
                         'spatial_gt': s_rel,               # Spatial Predicate GT: [71, M]
                         'contacting_gt': c_rel             # Contacting Predicate GT: [71, M]
                        }

                return entry
            elif self.mode == 'sgcls':
                if self.is_train:

                    FINAL_DISTRIBUTIONS = torch.softmax(self.fasterRCNN.RCNN_cls_score(FINAL_FEATURES)[:, 1:], dim=1)
                    FINAL_SCORES, PRED_LABELS = torch.max(FINAL_DISTRIBUTIONS, dim=1)
                    PRED_LABELS = PRED_LABELS + 1

                    union_boxes = torch.cat(
                        (im_idx[:, None], torch.min(FINAL_BBOXES[:, 1:3][pair[:, 0]], FINAL_BBOXES[:, 1:3][pair[:, 1]]),
                         torch.max(FINAL_BBOXES[:, 3:5][pair[:, 0]], FINAL_BBOXES[:, 3:5][pair[:, 1]])), 1)
                    union_feat = self.fasterRCNN.RCNN_roi_align(FINAL_BASE_FEATURES, union_boxes)
                    FINAL_BBOXES[:, 1:] = FINAL_BBOXES[:, 1:] / im_info[0, 2]
                    pair_rois = torch.cat((FINAL_BBOXES[pair[:, 0], 1:], FINAL_BBOXES[pair[:, 1], 1:]),
                                          1).data.cpu().numpy()
                    spatial_masks = torch.tensor(draw_union_boxes(pair_rois, 27) - 0.5).to(FINAL_FEATURES.device)

                    entry = {'boxes': FINAL_BBOXES,
                             'labels': FINAL_LABELS,  # here is the groundtruth
                             'scores': FINAL_SCORES,
                             'distribution': FINAL_DISTRIBUTIONS,
                             'base_feat': FINAL_BASE_FEATURES,
                             'pred_labels': PRED_LABELS,
                             'im_idx': im_idx,
                             'pair_idx': pair,
                             'human_idx': HUMAN_IDX,
                             'features': FINAL_FEATURES,
                             'union_feat': union_feat,
                             'union_box': union_boxes,
                             'spatial_masks': spatial_masks,
                             'attention_gt': a_rel,
                             'spatial_gt': s_rel,
                             'contacting_gt': c_rel}

                    return entry
                else:
                    FINAL_BBOXES[:, 1:] = FINAL_BBOXES[:, 1:] / im_info[0, 2]

                    FINAL_DISTRIBUTIONS = torch.softmax(self.fasterRCNN.RCNN_cls_score(FINAL_FEATURES)[:, 1:], dim=1)
                    FINAL_SCORES, PRED_LABELS = torch.max(FINAL_DISTRIBUTIONS, dim=1)
                    PRED_LABELS = PRED_LABELS + 1

                    entry = {'boxes': FINAL_BBOXES,                 # GT BBoxes: [109, 5]
                             'labels': FINAL_LABELS,                # GT Object label: [109]
                             'scores': FINAL_SCORES,                # Pred Object clf scores: [109]
                             'distribution': FINAL_DISTRIBUTIONS,   # Pred Object clf distribution: [109, 36]
                             'pred_labels': PRED_LABELS,            # Pred Object clf label: [109]
                             'im_idx': im_idx,                      # Im idx of pair: [71]
                             'pair_idx': pair,                      # Pair Idx: [71, 2]
                             'human_idx': HUMAN_IDX,                # Human Idx: [38, 1]
                             'features': FINAL_FEATURES,            # RoI features: [109, 2048]
                             'attention_gt': a_rel,                 # Attention Predicate GT: [71, M]
                             'spatial_gt': s_rel,                   # Spatial Predicate GT: [71, M]
                             'contacting_gt': c_rel,                # Contacting Predicate GT: [71, M]
                             'fmaps': FINAL_BASE_FEATURES,          # RoI base features: [38, 1024, 67, 38]
                             'im_info': im_info[0, 2]}

                    return entry

