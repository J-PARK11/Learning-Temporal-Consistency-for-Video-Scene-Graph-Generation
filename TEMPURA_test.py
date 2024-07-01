import os
import copy
import time
import torch
import debugpy
import datetime
import numpy as np
from tqdm import tqdm
np.set_printoptions(precision=4)    # 소수점 반올림 Setting
# debugpy.listen(("0.0.0.0", 5678))

from tools.utils.ds_track import get_sequence
from tools.utils.tempura_config import Config
from tools.utils.temporal_consistency import *
from tools.utils.object_detector import detector
from tools.utils.evaluation_recall import Get_AG_Evaluator

from dataloader.AG.action_genome import AG, cuda_collate_fn

from lib.tempura import TEMPURA                  

print('\n>>>>>>>>>>>>> AG TEMPURA Test.py <<<<<<<<<<<<<')
conf = Config()
gpu_device = torch.device('cuda:0')
log_val = open(conf.output_path+'log_val.txt', mode = 'w')
log_val.write('-'*30+'all_mode_eval'+'-'*30+'\n')

print('\n>>>>>>>>>>>>>>>> Dataset Load <<<<<<<<<<<<<<<<')
AG_dataset = AG(mode="test", datasize=conf.datasize, data_path=conf.data_path, filter_nonperson_box_frame=True,
                filter_small_box=False if conf.mode == 'predcls' else True)
dataloader = torch.utils.data.DataLoader(AG_dataset, shuffle=False, num_workers=0, collate_fn=cuda_collate_fn)

print('\n>>>>>>>>>>>> Object Detector Load <<<<<<<<<<<<')
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

print('\n>>>>>>>>>>>> Evaluator Definition <<<<<<<<<<<<')
eval_with, eval_semi, eval_no = Get_AG_Evaluator(mode=conf.mode, dataset=AG_dataset, output_path=conf.output_path)
print(f"AG VSGG Evaluator : With, Semi, No Constraint metrics used")

print('\n>>>>>>>>>>>> Inference Test Start <<<<<<<<<<<<')
start_time = time.time()
temp_cons_eval_spatial, temp_cons_eval_contact = torch.tensor([]), torch.tensor([])
with torch.no_grad():
    for b, data in enumerate(tqdm(dataloader)): 
        
        if b >= 10 :break
        
        im_data = copy.deepcopy(data[0].cuda(0))
        im_info = copy.deepcopy(data[1].cuda(0))
        gt_boxes = copy.deepcopy(data[2].cuda(0))
        num_boxes = copy.deepcopy(data[3].cuda(0))
        video_id = AG_dataset.valid_video_names[data[4]]
        gt_annotation = AG_dataset.gt_annotations[data[4]]
        
        entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)

        # Class Tracking: entry[indices] Only SGCLS & SGDET
        if conf.tracking:
            get_sequence(entry, gt_annotation, (im_info[0][:2]/im_info[0,2]).cpu().data,conf.mode)
        
        pred = model(entry, phase='test', unc=False)        
        
        eval_with.evaluate_scene_graph(gt_annotation, dict(pred))        
        eval_semi.evaluate_scene_graph(gt_annotation, dict(pred))
        eval_no.evaluate_scene_graph(gt_annotation, dict(pred))
        
        # Evaluate Temporal Consistency
        temp_cons_eval_spatial, temp_cons_eval_contact = evaluate_temp_cons(pred, temp_cons_eval_spatial, temp_cons_eval_contact, conf.mode)   
        
print('\n>>>>>>>>>>>>>> Inference Result <<<<<<<<<<<<<<')
total_time = time.time() - start_time
total_time_str = str(datetime.timedelta(seconds=int(total_time)))
print('Inference time {}'.format(total_time_str), flush=True)
print_temp_cons_score(temp_cons_eval_spatial, temp_cons_eval_contact, conf.mode)

constraint_type = 'With constraint'
print(f'\n====== {constraint_type} =====')
eval_with.print_stats(log_file=log_val, metric='Test')

constraint_type = 'Semi constraint'
print(f'\n====== {constraint_type} =====')
eval_semi.print_stats(log_file=log_val, metric='Test')

constraint_type = 'No constraint'
print(f'\n====== {constraint_type} =====')
eval_no.print_stats(log_file=log_val, metric='Test')

print('\n>>>>>>>>>> TEMPURA Test.py Complete <<<<<<<<<<')

