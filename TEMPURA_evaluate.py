import os
import cv2
import json
import copy
import time
import torch
import datetime
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
np.set_printoptions(precision=4)    # 소수점 반올림 Setting

from tools.utils.visualize import *
from tools.utils.ds_track import get_sequence
from tools.utils.tempura_config import Config
from tools.utils.object_detector import detector

from dataloader.AG.action_genome import AG, cuda_collate_fn

from lib.tempura import TEMPURA

print('\n>>>>>>>>>>> AG TEMPURA Evaluate.py <<<<<<<<<<<')
conf = Config()
gpu_device = torch.device('cuda:0')

print('\n>>>>>>>>>>>>>>>> Dataset Load <<<<<<<<<<<<<<<<')
AG_dataset = AG(mode="test", datasize=conf.datasize, data_path=conf.data_path, filter_nonperson_box_frame=True,
                filter_small_box=False if conf.mode == 'predcls' else True)
dataloader = torch.utils.data.DataLoader(AG_dataset, shuffle=False, num_workers=0, collate_fn=cuda_collate_fn)

print('\n>>>>>>>>>>>>> Object Detector Load <<<<<<<<<<<<<')
object_detector = detector(train=False, object_classes=AG_dataset.object_classes, use_SUPPLY=True, mode=conf.mode).to(device=gpu_device)
object_detector.eval()

print('\n>>>>>>>>>>>>>>>> TEMPURA Load <<<<<<<<<<<<<<<<')
model = TEMPURA(mode=conf.mode,
               attention_class_num=len(AG_dataset.attention_relationships),
               spatial_class_num=len(AG_dataset.spatial_relationships),
               contact_class_num=len(AG_dataset.contacting_relationships),
               obj_classes=AG_dataset.object_classes,
               enc_layer_num=conf.enc_layer,
               dec_layer_num=conf.dec_layer,
               obj_mem_compute = conf.obj_mem_compute,
               rel_mem_compute = conf.rel_mem_compute,
               take_obj_mem_feat= conf.take_obj_mem_feat,
               mem_fusion= conf.mem_fusion,
               selection = conf.mem_feat_selection,
               selection_lambda=conf.mem_feat_lambda,
               obj_head = conf.obj_head,
               rel_head = conf.rel_head,
               K = conf.K,
               tracking= conf.tracking
            ).to(device=gpu_device)
model.eval()
ckpt = torch.load(conf.model_path, map_location=gpu_device)
model.load_state_dict(ckpt['state_dict'], strict=True)
print('CKPT {} is loaded'.format(conf.model_path))

print('\n>>>>>>>>>>>>>> Visualize Start <<<<<<<<<<<<<<')
log = None
start_time = time.time()
with torch.no_grad():
    for b, data in enumerate(tqdm(dataloader)): 
        
        if b>10:break
        
        im_data = copy.deepcopy(data[0].cuda(0))
        im_info = copy.deepcopy(data[1].cuda(0))
        gt_boxes = copy.deepcopy(data[2].cuda(0))
        num_boxes = copy.deepcopy(data[3].cuda(0))
        gt_annotation = AG_dataset.gt_annotations[data[4]]
        
        entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)
        entry['video_id'] = AG_dataset.valid_video_names[data[4]]
        entry['video_size'] = AG_dataset.video_size[data[4]]

        if conf.tracking:
            get_sequence(entry, gt_annotation, (im_info[0][:2]/im_info[0,2]).cpu().data,conf.mode)
        
        pred = model(entry, phase='test', unc=False)
        
        viz_img = AG_dataset.get_viz_img(b)
        # log = Viz_GT_Scene_Graph(viz_img, conf.output_path, gt_annotation, entry, log)
        # log = Viz_Pred_Scene_Graph(viz_img, conf.output_path, pred, log)
        # log = Viz_BBox_in_Scene(viz_img, conf.output_path, gt_annotation, entry)
        # log = Viz_Just_Scene(viz_img, conf.output_path, entry)
        # log = Viz_Object_TSNE(conf.output_path, pred)
        # log = Viz_Temporal_Consistency(conf.output_path, pred)
        
print('\n>>>>>>>>>>>>>> Inference Result <<<<<<<<<<<<<<')
if log != None: log.close()
total_time = time.time() - start_time
total_time_str = str(datetime.timedelta(seconds=int(total_time)))
print('Inference time {}'.format(total_time_str), flush=True)

print('\n>>>>>>>> TEMPURA Evaluate.py Complete <<<<<<<<')
