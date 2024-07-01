import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.manifold import TSNE

import warnings
warnings.filterwarnings(action='ignore')

AG_obj_label = ['__background__', 'person', 'bag', 'bed', 'blanket', 'book', 'box', 'broom', 'chair', 'closet/cabinet', 'clothes', 'cup/glass/bottle', 'dish', 'door', 'doorknob', 'doorway', 'floor', 'food', 'groceries', 'laptop', 'light', 'medicine', 'mirror', 'paper/notebook', 'phone/camera', 'picture', 'pillow', 'refrigerator', 'sandwich', 'shelf', 'shoe', 'sofa/couch', 'table', 'television', 'towel', 'vacuum', 'window']
AG_att_label = ['looking_at', 'not_looking_at', 'unsure']
AG_spatial_label = ['above', 'beneath', 'in_front_of', 'behind', 'on_the_side_of', 'in']
AG_contact_label = ['carrying', 'covered_by', 'drinking_from', 'eating', 'have_it_on_the_back', 'holding', 'leaning_on', 'lying_on', 'not_contacting', 'other_relationship', 'sitting_on', 'standing_on', 'touching', 'twisting', 'wearing', 'wiping', 'writing_on']

colors_tab20b = plt.cm.tab20b.colors
colors_tab20c = plt.cm.tab20c.colors
colors_comb = colors_tab20b + colors_tab20c
marker_dict = ['o', 'x', 'D', '+', 'p', '*', 's', '1', '_']

def check_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return None

def draw_rectangle(image, bbox, color=(255, 0, 0), thickness=2):
    x1, y1, x2, y2 = bbox
    image_copy = image.copy()
    start_point = (x1, y1)
    end_point = (x2, y2)
    image_with_rectangle = cv2.rectangle(image_copy, start_point, end_point, color, thickness)
    return image_with_rectangle

def close_palette():
    plt.cla()
    plt.clf()
    plt.close()
    return None

def dir_setting(path, entry):
    video_id = entry['video_id'][:-4]
    video_size = entry['video_size']
    video_path = os.path.join(path, video_id)
    check_dir(path)
    check_dir(video_path)
    return path, video_id, video_size, video_path
    

# =========================== Main Module =========================== #

# GT 장면 그래프 생성 시각화.
def Viz_GT_Scene_Graph(viz_img, path, gt, entry, log, axis=False):
    path = os.path.join(path, 'gt')
    path, video_id, video_size, video_path = dir_setting(path, entry)
    
    if log == None:
        log = open(path+'/log.txt', mode = 'w')
        log.write('-'*15+'GT Scene Graph'+'-'*15+'\n')
    
    log.write(f'[{video_id}]: {viz_img.shape[0]} frames =================================================================\n')
    for f_id, frame in enumerate(viz_img):
        plt.figure(figsize=(10,10))
        palette = (cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).clip(0,255)
        person_bbox = gt[f_id][0]['person_bbox'][0].round().astype(int)
        palette = draw_rectangle(palette, person_bbox, color=(255,0,0)) # red

        for o_id in range(1, len(gt[f_id])):
            obj_bbox = gt[f_id][o_id]['bbox'].round().astype(int)
            palette = draw_rectangle(palette, obj_bbox, color=(0,0,255)) # blue
            Get_GT_Scene_Graph(gt, f_id, o_id, video_size, log)
        log.write('\n')
        plt.axis(axis)
        plt.imshow(palette)
        plt.savefig(os.path.join(video_path, f'{f_id:03d}.png'), bbox_inches='tight')
        close_palette()

    return log
    
def Get_GT_Scene_Graph(gt, f_id, o_id, video_size, log):
    obj, att, spatial, contact = AG_obj_label[gt[f_id][o_id]['class']], AG_att_label[gt[f_id][o_id]['attention_relationship'][0]], \
                                 AG_spatial_label[gt[f_id][o_id]['spatial_relationship'][0]], AG_contact_label[gt[f_id][o_id]['contacting_relationship'][0]]
    log_f_id, log_obj, log_att, log_spatial, log_contact = str(f_id).rjust(3, ' '), obj.rjust(18,' '), att.rjust(15,' '), spatial.rjust(15,' '), contact.rjust(20,' ')
    log.write(f'{log_f_id}f, {log_obj}: [ {log_att}, {log_spatial}, {log_contact} ]\n')
    x, y = 0, (video_size[1] + o_id*15)
    plt.text(x,y,f'Person <{att} & {spatial} & {contact}> {obj}', fontsize=12, color='black')
                                
    return None
    
# Pred 장면 그래프 생성 시각화.
def Viz_Pred_Scene_Graph(viz_img, path, pred, log, axis=False):
    path = os.path.join(path, 'pred')
    path, video_id, video_size, video_path = dir_setting(path, pred)
    
    start_idx, end_idx = 0, 0
    pred_boxes = pred['boxes']
    frame_idx = np.array(pred_boxes[:,0].cpu())
    
    if log == None:
        log = open(path+'/log.txt', mode = 'w')
        log.write('-'*15+'Pred Scene Graph'+'-'*15+'\n')
    
    log.write(f'[{video_id}]: {viz_img.shape[0]} frames =================================================================\n')
    for f_id, frame in enumerate(viz_img):
        target_bool = np.where(frame_idx == f_id)[0]
        target_bbox = pred_boxes[target_bool, :].cpu().numpy()[:,1:].round().astype(int)
        end_idx = (start_idx + len(target_bool) -1)
        
        obj_pred = pred['pred_labels'][target_bool].cpu()
        att_pred = torch.argmax(pred['attention_distribution'][start_idx:end_idx], dim=1).cpu()
        spatial_pred = torch.argmax(pred['spatial_distribution'][start_idx:end_idx], dim=1).cpu()
        contact_pred = torch.argmax(pred['contacting_distribution'][start_idx:end_idx], dim=1).cpu()
        pred_dict = {'obj':obj_pred, 'att':att_pred, 'spatial':spatial_pred, 'contact':contact_pred}
        
        plt.figure(figsize=(10,10))
        palette = (cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).clip(0,255)

        for o_id, bbox in enumerate(target_bbox):
            if o_id == 0:   # Person
                palette = draw_rectangle(palette, bbox, color=(255,0,0)) # Red
            else:           # Object
                palette = draw_rectangle(palette, bbox, color=(0,255,0)) # Green
                Get_Pred_Scene_Graph(pred_dict, f_id, o_id, video_size, log)
        
        log.write('\n')
        plt.axis(axis)
        plt.imshow(palette)
        plt.savefig(os.path.join(video_path, f'{f_id:03d}.png'), bbox_inches='tight')
        close_palette()
        start_idx = end_idx

    return log

# Pred 장면 그래프 생성 시각화.
def Log_KCC_Data_Prediction(viz_img, video_name, pred, log):
    video_log = dict()
    len_img = viz_img.shape[0]
    im_idx = pred['im_idx']
    
    try:    
        obj_pred = pred['obj_class']
    except:
        obj_pred = pred['pred_labels']
        obj_pred = obj_pred[obj_pred!=1]
        
    att_pred = torch.argmax(pred['attention_distribution'], 1)
    spatial_pred = torch.argmax(pred['spatial_distribution'], 1)
    contact_pred = torch.argmax(pred['contacting_distribution'], 1)

    for i in range(len_img):
        frame_log = dict()
        target_idx = (im_idx == i)
        
        obj_list = obj_pred[target_idx]
        att_list = att_pred[target_idx]
        spatial_list = spatial_pred[target_idx]
        contact_list = contact_pred[target_idx]
        
        for j, (o, a, s, c) in enumerate(zip(obj_list, att_list, spatial_list, contact_list)):            
            obj_log = dict()
            
            o, a, s, c = AG_obj_label[o], AG_att_label[a], AG_spatial_label[s], AG_contact_label[c]
            
            obj_log['subject'] = 'person'
            obj_log['attention_rel'] = a
            obj_log['spatial_rel'] = s
            obj_log['contact_rel'] = c
            obj_log['object'] = o
            
            frame_log[f'triplet_{(j+1):02d}'] = obj_log
        video_log[f'frame_{(i+1):04d}'] = frame_log
        
    log[video_name] = video_log
    return log

def Get_Pred_Scene_Graph(pred_dict, f_id, o_id, video_size, log):
    obj, att, spatial, contact = AG_obj_label[pred_dict['obj'][o_id]], AG_att_label[pred_dict['att'][o_id-1]], \
                                 AG_spatial_label[pred_dict['spatial'][o_id-1]], AG_contact_label[pred_dict['contact'][o_id-1]]
    log_obj, log_att, log_spatial, log_contact = obj.rjust(18,' '), att.rjust(15,' '), spatial.rjust(15,' '), contact.rjust(20,' ')
    log.write(f'{f_id}f, {log_obj}: [ {log_att}, {log_spatial}, {log_contact} ]\n')
    x, y = 0, (video_size[1] + o_id*15)
    plt.text(x,y,f'Person <{att} & {spatial} & {contact}> {obj}', fontsize=12, color='black')
                                
    return None

# Only BBox 장면 시각화.
def Viz_BBox_in_Scene(viz_img, path, gt, entry, axis=False):
    path = os.path.join(path, 'bbox')
    path, video_id, video_size, video_path = dir_setting(path, entry)
    
    for f_id, frame in enumerate(viz_img):
        plt.figure(figsize=(10,10))
        palette = (cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).clip(0,255)
        person_bbox = gt[f_id][0]['person_bbox'][0].round().astype(int)
        palette = draw_rectangle(palette, person_bbox, color=(255,0,0)) # red

        for o_id in range(1, len(gt[f_id])):
            obj_bbox = gt[f_id][o_id]['bbox'].round().astype(int)
            palette = draw_rectangle(palette, obj_bbox, color=(0,0,255)) # blue
        plt.axis(axis)
        plt.imshow(palette)
        plt.savefig(os.path.join(video_path, f'{f_id:03d}.png'), bbox_inches='tight')
        close_palette()

    return None

# Raw Frame 시각화.
def Viz_Just_Scene(viz_img, path, entry, axis=False):
    path = os.path.join(path, 'frame')
    path, video_id, video_size, video_path = dir_setting(path, entry)
    
    for f_id, frame in enumerate(viz_img):
        plt.figure(figsize=(10,10))
        palette = (cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).clip(0,255)
        plt.axis(axis)
        plt.imshow(palette)
        plt.savefig(os.path.join(video_path, f'{f_id:03d}.png'), bbox_inches='tight')
        close_palette()

    return None

# 객체 별 T-SNE 분포 시각화.
def Viz_Object_TSNE(path, pred, axis=True):
    path = os.path.join(path, 'TSNE')
    path, video_id, video_size, video_path = dir_setting(path, pred)
    
    tsne = TSNE(n_components=2, random_state=0)
    c_tsne_vec = tsne.fit_transform(pred['contacting_distribution'].cpu())
    
    obj_label = pred['pred_labels'][pred['pred_labels']!=1]
    unique_obj = torch.unique(obj_label)
    
    for o_id, target in enumerate(unique_obj):
        target_bool = (obj_label==target)
        target_obj = AG_obj_label[target].replace('/','_')
        pred_c_target = torch.argmax(pred['contacting_distribution'][target_bool], 1).cpu().detach()
        pred_c_unique = torch.unique(pred_c_target)
        c_color = np.array(colors_comb)[pred_c_unique].reshape(-1,3)
        c_tsne_target = c_tsne_vec[target_bool.cpu().detach()]
        
        plt.figure(figsize=(10,10))
        for rel_id, rel in enumerate(pred_c_unique):
            try:
                c_rel_target = c_tsne_target[pred_c_target == rel]
            except:
                c_rel_target = c_tsne_target[[pred_c_target == rel]]
            plt.scatter(c_rel_target[:,0], c_rel_target[:,1], marker=marker_dict[rel_id], color=c_color[rel_id], label=AG_contact_label[rel])
        plt.legend(loc='best')
        plt.axis(axis)
        plt.savefig(os.path.join(video_path, f'{target_obj}.png'), bbox_inches='tight')
        close_palette()
    
    return None

# 객체별 시간적 일관성 경향 파악.
def Viz_Temporal_Consistency(path, pred, axis=True):
    path = os.path.join(path, 'temporal_consistency')
    path, video_id, video_size, video_path = dir_setting(path, pred)
    
    contact_gt = torch.tensor([i[0] for i in pred['contacting_gt']])
    contact_pred = pred['contacting_distribution']
    
    obj_label = pred['pred_labels'][pred['pred_labels']!=1]
    unique_obj = torch.unique(obj_label)
    
    for o_id, target in enumerate(unique_obj):
        target_bool = (obj_label==target)
        target_obj = AG_obj_label[target].replace('/','_')
        c_target_gt = contact_gt[target_bool]
        c_target_gt = F.one_hot(c_target_gt, 17)
        c_target_pred = contact_pred[target_bool]
        
        Viz_Logit_Matrix(video_path, target_obj, c_target_gt, c_target_pred, axis)
    
    return None
        
def Viz_Logit_Matrix(video_path, target, gt, pred, axis):
    fig, ax = plt.subplots(1, 2, figsize=(15, 15))
    ax[0].matshow(gt.cpu().detach().numpy())
    ax[1].matshow(pred.cpu().detach().numpy())
    ax[0].set_xlabel('Label')
    ax[1].set_xlabel('Label')
    ax[0].set_ylabel('#frame')
    ax[1].set_ylabel('#frame')
    plt.axis(axis)
    plt.savefig(os.path.join(video_path, f'{target}.png'), bbox_inches='tight')
    plt.cla()   # clear the current axes
    plt.clf()   # clear the current figure
    
    
    
    
    
