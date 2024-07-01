import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device('cuda:0')

def find_consecutive_duplicates(target_bool, gt_tensor, pred_tensor, window=6):
    consecutive_itv = []
    consecutive_cnt = 0
    prev_state = -1
                                             
    for id, (bool, gt, pred) in enumerate(zip(target_bool, gt_tensor, pred_tensor)):
        if (bool == True) and (gt == prev_state):
            consecutive_cnt += 1
        else:
            if consecutive_cnt >= window:
                consecutive_itv.append([id-consecutive_cnt, id])
            consecutive_cnt = 0
            prev_state = gt
    
    if (bool==True) and (gt == prev_state) and (consecutive_cnt >= window):
        consecutive_itv.append([id-consecutive_cnt, id])
                            
    return consecutive_itv

def evaluate_temp_cons(pred, temp_cons_eval_spatial, temp_cons_eval_contact, mode):
    
    if mode == 'sgdet': return None, None
    
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    video_spatial_cons, video_contact_cons = torch.tensor([]), torch.tensor([])
    
    spatial_gt_tensor = torch.tensor([i[0] for i in pred['spatial_gt']])
    contact_gt_tensor = torch.tensor([i[0] for i in pred['contacting_gt']])
    spatial_pred_tensor = pred['spatial_distribution']
    contact_pred_tensor = pred['contacting_distribution']
    
    obj_indices = pred['pred_labels']!=1
    obj_cls = pred['pred_labels'][obj_indices]
    unique_obj_cls = torch.unique(obj_cls)
    
    for cls in unique_obj_cls:
        
        target_bool = (obj_cls==cls)            
        
        # Spatial Score Compute
        consecutive_itv = find_consecutive_duplicates(target_bool, spatial_gt_tensor, spatial_pred_tensor)
        for s_idx, e_idx in consecutive_itv:
            spatial_gt = spatial_gt_tensor[s_idx:e_idx]
            spatial_gt = F.one_hot(spatial_gt, 6)
            spatial_pred = spatial_pred_tensor[s_idx:e_idx]
            
            spatial_p, spatial_q = F.log_softmax(spatial_gt.type(torch.float32), dim=1).to(device), F.softmax(spatial_pred, dim=1)
            spatial_score = torch.tensor([kl_loss(spatial_p, spatial_q)])
            video_spatial_cons = torch.cat([video_spatial_cons, spatial_score])
        
        # Contact Score Compute
        consecutive_itv = find_consecutive_duplicates(target_bool, contact_gt_tensor, contact_pred_tensor)
        for s_idx, e_idx in consecutive_itv:

            contact_gt = contact_gt_tensor[s_idx:e_idx]
            contact_gt = F.one_hot(contact_gt, 17)
            contact_pred = contact_pred_tensor[s_idx:e_idx]
            
            contact_p, contact_q = F.log_softmax(contact_gt.type(torch.float32), dim=1).to(device), F.softmax(contact_pred, dim=1)
            contact_score = torch.tensor([kl_loss(contact_p, contact_q)])
            video_contact_cons = torch.cat([video_contact_cons, contact_score])
    
    temp_cons_eval_spatial = torch.cat([temp_cons_eval_spatial, video_spatial_cons])
    temp_cons_eval_contact = torch.cat([temp_cons_eval_contact, video_contact_cons])
    
    return temp_cons_eval_spatial, temp_cons_eval_contact

def print_temp_cons_score(temp_cons_eval_spatial, temp_cons_eval_contact, mode):
    if mode != 'sgdet':
        num_s_itv, num_c_itv = len(temp_cons_eval_spatial), len(temp_cons_eval_contact)
        s_score, c_score = temp_cons_eval_spatial.mean()*100, temp_cons_eval_contact.mean()*100
        t_score = ((s_score + c_score) / 2)

        print(f'Spatial Temporal Consistency Score: {s_score:.6f}, {num_s_itv} Intervals')
        print(f'Contacting Temporal Consistency Score: {c_score:.6f}, {num_c_itv} Intervals')
        print(f'Temporal Consistency Score: {t_score:.6f}')