# Common Library
import json
import math
import torch
import itertools
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F

from tools.TokenGT.tokengt.models.tokengt import TokenGTModel, TokenGTEncoder
from tools.utils.object_classifier import ObjectClassifier
from tools.utils.word_vectors import obj_edge_vectors

import dgl
import dgl.function as fn
from dgl.nn import GlobalAttentionPooling
import networkx as nx

from graph_transformer_pytorch import GraphTransformer

import warnings
warnings.filterwarnings('ignore')
device = torch.device('cuda:0')

class TEAT_GT(nn.Module):

    def __init__(self, mode='predcls', attention_class_num=None, spatial_class_num=None, \
                 contact_class_num=None, obj_classes=None, tracking=None, args=None):

        super(TEAT_GT, self).__init__()
        
        self.obj_classes = obj_classes
        self.attention_class_num = attention_class_num
        self.spatial_class_num = spatial_class_num
        self.contact_class_num = contact_class_num
        
        assert mode in ('sgdet', 'sgcls', 'predcls')
        self.mode = mode
        self.tracking = tracking
        
        # OSPU
        self.object_classifier = ObjectClassifier(mode=self.mode, obj_classes=self.obj_classes, 
                                                obj_head = "linear", mem_compute=None, K=4, selection=None,
                                                selection_lambda=None, tracking=self.tracking)

        # RPN Feature Embedding Module
        self.subj_fc = nn.Linear(2048, 968)
        self.obj_fc = nn.Linear(2048, 968)
        
        # Object Label Tokenizer
        embed_vecs = obj_edge_vectors(obj_classes, wv_type='glove.6B', wv_dir='tools/utils/word_vectors', wv_dim=200)
        embed_vecs_union = embed_vecs.repeat(2,1)
        
        self.node_label_tokenizer = nn.Embedding(len(obj_classes), 200)
        self.node_label_tokenizer.weight.data = embed_vecs.clone()
        
        # TokenGT
        self.args = args
        self.TokenGT_encoder = TokenGTEncoder(self.args)
        self.TokenGT_model = TokenGTModel(self.args, self.TokenGT_encoder)

        # GAT
        self.gat = GraphTransformer(
                dim = 10,
                depth = 4,
                edge_dim = 1,           
                with_feedforwards = True,   # whether to add a feedforward after each attention layer, suggested by literature to be needed
                gated_residual = True,      # to use the gated residual to prevent over-smoothing
                rel_pos_emb = True          # set to True if the nodes are ordered, default to False
            ).to(device)
        
        self.gat_semantic = GraphTransformer(
                dim = 768,
                depth = 4,
                edge_dim = 1,           
                with_feedforwards = True,   # whether to add a feedforward after each attention layer, suggested by literature to be needed
                gated_residual = True,      # to use the gated residual to prevent over-smoothing
                rel_pos_emb = True          # set to True if the nodes are ordered, default to False
            ).to(device)
        
        self.gate_nn = nn.Linear(10, 1)
        self.gap = GlobalAttentionPooling(self.gate_nn).to(device)
        self.gate_sem_nn = nn.Linear(768, 1)
        self.gap_sem = GlobalAttentionPooling(self.gate_sem_nn).to(device)
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

        self.soft_max = nn.Softmax(-1)
        self.sigmoid = nn.Sigmoid()
        
        # Graph GlobalAttentionPooling
        self.gate_gru_nn = nn.Linear(768, 1)
        self.gap_gru = GlobalAttentionPooling(self.gate_gru_nn)

        self.viz_idx = 0
        
    def forward(self, entry, phase='train',unc=False):

        # OSPU
        entry = self.object_classifier(entry, phase=phase, unc=unc)   
        
        # ***************** Page 1 : Faster RCNN to Node ***************** #
        max_im_idx = int(torch.max(entry['im_idx']))
        whole_im_idx = entry['im_idx'].type(torch.int)        
        
        person_idx, person_im_idx = [], []
        for find_idx in range(max_im_idx+1):  
            try:
                ele_idx = (whole_im_idx==find_idx).nonzero().flatten()[0]
                person_idx.append(int(ele_idx))
                person_im_idx.append(int(find_idx))
            except:
                pass
        person_idx, person_im_idx = torch.tensor(person_idx).to(device), torch.tensor(person_im_idx).to(device)    
        
        # FasterRCNN RPN Feature 가져오기
        subj_rep = entry['features'][entry['pair_idx'][:, 0]]       # [71, 2048]: Subject Backbone Features
        subj_rep = subj_rep[person_idx]                             # [71-A, 2048]: Subject Backbone Features
        obj_rep = entry['features'][entry['pair_idx'][:, 1]]        # [71, 2048]: Object Backbone Features

        # Person과 Object Independent RPN Node Feature Embedding
        subj_rep = self.subj_fc(subj_rep)                               # [71-A, 768]
        obj_rep = self.obj_fc(obj_rep)                                  # [71, 768]
        
        # BBox Coordinate Embedding        
        subj_bbox = entry['boxes'][entry['pair_idx'][:, 0]][:,1:]      
        obj_bbox = entry['boxes'][entry['pair_idx'][:, 1]][:,1:]  
        subj_bbox = subj_bbox[person_idx]
        
        # Object Label Embedding
        subj_class = entry['pred_labels'][entry['pair_idx'][:, 0]]      # [71]
        obj_class = entry['pred_labels'][entry['pair_idx'][:, 1]]       # [71]
        subj_class = subj_class[person_idx]                             # [71-A]
        
        subj_class_token = self.node_label_tokenizer(subj_class)                              # [71-A, 200]
        obj_class_token = self.node_label_tokenizer(obj_class)                                # [71, 200]
                
        # Composition Token Definition: Object Node = Edge, Total = Person + Object + Edge
        person_token = torch.cat((subj_rep, subj_class_token), 1)                 # [71-A, 1168] (P_Node)
        obj_token = torch.cat((obj_rep, obj_class_token), 1)                       # [71, 1168] (O_Node)
        
        # Total Token Definition
        total_token = torch.cat((person_token, obj_token), dim=0)                             # [71 + 71-A, 1168]
        token_im_idx = torch.cat([person_im_idx, entry['im_idx']])                            # [71 + 71-A]
        token_bbox = torch.cat([subj_bbox, obj_bbox], dim=0)                                  # [71 + 71-A, 4]
        token_bbox_center = torch.tensor([[box[[0,2]].mean(), box[[1,3]].mean()] for box in token_bbox]).to(token_im_idx.device)    # [71 + 71-A, 2]        
        
        # Graph Definition for TokenGT
        idx = entry['im_idx'].unique().type(torch.int)  # 0~Franme
        token_num, node_num = total_token.shape[0], torch.bincount(token_im_idx.type(torch.int)) # Num of total node, Num of node by frames
        token_order = torch.cat([(token_im_idx==int(i)).nonzero().squeeze() for i in idx])       # Token Index ordered by frame
        
        # Token Ordering [(P,O), Embed]----
        total_token = total_token[token_order]                   
        token_im_idx = token_im_idx[token_order].type(torch.int) 
        token_bbox_center = token_bbox_center[token_order]       
        
        # ***************** Page 2 : Video clipping ***************** #
        clip_token, clip_im_idx, clip_bbox_center = [], [], []
        clip_size = 5
        clip_num = math.ceil(int(max(token_im_idx)+1)/clip_size)
        for i in range(clip_num):
            start_idx, end_idx = i*clip_size, (i+1)*clip_size
            clip_idx = (start_idx <= token_im_idx) & (token_im_idx < end_idx)
            clip_token_ele = total_token[clip_idx]
            clip_im_idx_ele = token_im_idx[clip_idx]
            clip_bbox_center_ele = token_bbox_center[clip_idx]
            clip_token.append(clip_token_ele), clip_im_idx.append(clip_im_idx_ele), clip_bbox_center.append(clip_bbox_center_ele)

        # **************************************************************** #
    
        # ***************** Page 3 : Graph Representation ***************** #        
        spatial_thr = 0.5
        edge_thr = np.round(np.sqrt(entry['video_size'][0]**2 + entry['video_size'][1]**2)*spatial_thr, 4)  # (PredCLS, SGCLS: 0.9), (Low Thr, SGDET: 0.5)
        sim_thr = 0.75
        
        global_output, frame_kl_div, semantic_kl_div = [], [], []
        hidden_state = None
        clip_iter = 0
        for clip_token_ele, clip_im_idx_ele, clip_bbox_center_ele in zip(clip_token, clip_im_idx, clip_bbox_center):
            start_idx, end_idx = clip_iter*clip_size, int(max(clip_im_idx_ele)) + 1
            past_im_idx, edge_ele_num = 0, 0
            past_state, current_state = {}, {}        
            spatial_u, spatial_v, edge_num, edge_feat = [], [], [], []
            org_u, org_v = [], []
                    
            for i in range(start_idx, end_idx, 1):
                u, v = [], []
                node_idx = (clip_im_idx_ele == i).nonzero().flatten()
                node_bbox = clip_bbox_center_ele[node_idx]
                node_rpn = clip_token_ele[node_idx]
                
                if len(node_idx) == 0:
                    zip_dict = {}           
                else:
                    zip_dict = {int(key): [bbox, rpn, int(org_node_idx)] for key, bbox, rpn, org_node_idx in zip(node_idx - min(node_idx), node_bbox, node_rpn, node_idx)}      
                    
                # Spatial Edge Definition: Proximity
                ncr = list(itertools.combinations(zip_dict, 2))
                for ncr_u, ncr_v in ncr:
                    dist = torch.sqrt((zip_dict[ncr_u][0][0]-zip_dict[ncr_v][0][0])**2 + (zip_dict[ncr_u][0][1]-zip_dict[ncr_v][0][1])**2)
                    if dist <= edge_thr:
                        org_u.append(zip_dict[ncr_u][2]), org_v.append(zip_dict[ncr_v][2])
                        org_u.append(zip_dict[ncr_v][2]), org_v.append(zip_dict[ncr_u][2])
                        u.append(ncr_u), v.append(ncr_v)
                        u.append(ncr_v), v.append(ncr_u)
                        edge_feat.append(0), edge_feat.append(0)
                        edge_ele_num += 2
                
                # Temporal Edge Definition: RPN Feature Similarity
                temp_ncr = list(itertools.product(past_state, zip_dict))
                for ncr_u, ncr_v in temp_ncr:
                    u_rpn, v_rpn = past_state[ncr_u][1], zip_dict[ncr_v][1]
                    cos_sim = torch.dot(u_rpn, v_rpn) / (torch.norm(u_rpn) * torch.norm(v_rpn))
                    
                    if cos_sim >= sim_thr:
                        org_u.append(past_state[ncr_u][2]), org_v.append(zip_dict[ncr_v][2])
                        org_u.append(zip_dict[ncr_v][2]), org_v.append(past_state[ncr_u][2])
                        edge_feat.append(1), edge_feat.append(1)
                        edge_ele_num += 2
                
                past_state = zip_dict
                spatial_u.append(u), spatial_v.append(v)
                edge_num.append(edge_ele_num)
                edge_ele_num = 0

            # For low spatial edge Threshold & SGDET
            if len(org_u) == 0:
                org_u.append(0), org_v.append(1), edge_feat.append(0)
                org_u.append(1), org_v.append(0), edge_feat.append(0)
                u.append(ncr_u), v.append(ncr_v)
                u.append(ncr_v), v.append(ncr_u)
                edge_num.append(2)
            
            org_u = torch.tensor(org_u).to(device)
            org_v = torch.tensor(org_v).to(device)
            edge_num = torch.tensor(edge_num).to(device)
            edge_index = torch.cat((org_u.unsqueeze(0), org_v.unsqueeze(0)), dim=0)
            edge_data = torch.tensor(edge_feat).type(torch.int).to(device)
                
            # Graph Matrix
            g = dgl.DGLGraph().to(device)
            g.add_nodes(len(clip_im_idx_ele))
            g.add_edges(org_u, org_v)
        
            # Laplacian
            A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float).toarray()
            N = np.diag(g.in_degrees().cpu().clip(1) ** -0.5)
            L = np.eye(g.number_of_nodes()) - N @ A @ N

            # (sorted) eigenvectors with numpy
            EigVal, EigVec = np.linalg.eigh(L)
            EigVal, EigVec = torch.tensor(EigVal).type(torch.float32).to(device), torch.tensor(EigVec).type(torch.float32).to(device)
                
            # Batched Data
            node_per_frame = node_num[start_idx:end_idx]
            batched_data = dict()
            batched_data['idx'] = None                 
            batched_data['node_data'] = clip_token_ele
            batched_data['node_num'] = sum(node_num[start_idx:end_idx]).unsqueeze(0)
            batched_data['in_degree'] = g.in_degrees()
            batched_data['out_degree'] = g.out_degrees()
        
            batched_data['lap_eigvec'] = EigVec
            batched_data['lap_eigval'] = EigVal
            batched_data['temporal_pe'] = clip_im_idx_ele.unsqueeze(1)
            
            batched_data['edge_index'] = edge_index
            batched_data['edge_data'] = edge_data.unsqueeze(1)
            batched_data['edge_num'] = sum(edge_num).unsqueeze(0)
            
            batched_data['node_per_frame'] = node_per_frame
            batched_data['hidden_state'] = hidden_state
            
            # **************************************************************** #
            
            # ***************** Page 3 : TokenGT ***************** #
            global_output_ele, attn_dict, hidden_x = self.TokenGT_model(batched_data)
            hidden_state = self.gap_gru(g, hidden_x)
            global_output.append(global_output_ele)
            clip_iter += 1
            
            # GAT
            if phase=='train':
                lap_node_id_k = 10
                graph_symbol_list, graph_sementic_list = [], []
                frame_start = start_idx-start_idx
                frame_end = end_idx-start_idx
                savor = 0
                for i in range(frame_start, frame_end, 1):
                    g_frame = dgl.DGLGraph().to(device)
                    g_frame.add_nodes(int(node_per_frame[i]))
                    g_frame.add_edges(spatial_u[i], spatial_v[i])
                    
                    A = g_frame.adjacency_matrix_scipy(return_edge_ids=False).astype(float).toarray()
                    N = np.diag(g_frame.in_degrees().cpu().clip(1) ** -0.5)
                    L = np.eye(g_frame.number_of_nodes()) - N @ A @ N
                    
                    EigVal_frame, EigVec_frame = np.linalg.eigh(L)
                    EigVal_frame, EigVec_frame = torch.tensor(EigVal_frame).type(torch.float32).to(device), torch.tensor(EigVec_frame).type(torch.float32).to(device)
                    
                    lap_dim = EigVec_frame.size(-1)
                    if lap_node_id_k > lap_dim:
                        eigvec = EigVec_frame.repeat(1,int(lap_node_id_k/2))[:,:lap_node_id_k]
                    else:
                        eigvec = EigVec_frame[:, :lap_node_id_k]
                    
                    nodes = eigvec.unsqueeze(0).to(device).type(torch.float32)
                    edges = torch.tensor(A.reshape(1,lap_dim,lap_dim,-1)).to(device).type(torch.float32)
                    
                    next_savor = savor + node_per_frame[i]
                    node_semantic = hidden_x[savor:next_savor]
                    next_savor = savor
                    
                    node_output, _ = self.gat(nodes, edges)
                    node_semantic_output, _ = self.gat_semantic(node_semantic.unsqueeze(0), edges)
                    
                    graph_symbol = self.gap(g_frame, node_output.squeeze())
                    graph_semantic_symbol = self.gap_sem(g_frame, node_semantic_output.squeeze())
                    
                    graph_symbol_list.append(graph_symbol)
                    graph_sementic_list.append(graph_semantic_symbol)
                
                ncr = list(itertools.combinations(np.arange(frame_start, frame_end), 2))
                for u, v in ncr:
                    p, q = F.log_softmax(graph_symbol_list[u]), F.softmax(graph_symbol_list[v])
                    p_sem, q_sem = F.log_softmax(graph_sementic_list[u]), F.softmax(graph_sementic_list[v])
                    score = self.kl_loss(p, q)/(v-u)
                    sem_score = self.kl_loss(p_sem, q_sem)/(v-u)
                    if score >= 0:
                        frame_kl_div.append(score)
                    if sem_score >= 0:
                        semantic_kl_div.append(sem_score)
        
        global_output = torch.cat(global_output, 0)   
        
        att_dist = global_output[:, :3]
        spatial_dist = global_output[:, 3:9]
        contact_dist = global_output[:, 9:]
        
        att_dist = self.soft_max(att_dist)           # self.softmax(att_dist)
        spatial_dist = self.sigmoid(spatial_dist)
        contact_dist = self.sigmoid(contact_dist)
        
        entry["attention_distribution"] = att_dist      # [71, 3]
        entry["spatial_distribution"] = spatial_dist    # [71, 6]
        entry["contacting_distribution"] = contact_dist # [71, 17]
        
        entry['structure_temp_loss'] = torch.tensor(frame_kl_div).to(device).type(torch.float32)
        entry['semantic_temp_loss'] = torch.tensor(semantic_kl_div).to(device).type(torch.float32)  
              
        # **************************************************************** #
        
        return entry

