import os
import time
import copy
import torch
import pickle
import debugpy
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from torch import optim
import pytorch_warmup as warmup
np.set_printoptions(precision=4)
debugpy.listen(("0.0.0.0", 5678))

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR
from pytorch_metric_learning import losses as metric_loss

from tools.utils.Memory import *
from tools.utils.infoNCE import * 
from tools.utils.AdamW import AdamW
from tools.utils.Uncertainty import *
from tools.utils.tempura_config import Config
from tools.utils.ds_track import get_sequence
from tools.utils.env import set_seed_and_igwarn, set_train_dir
from tools.utils.object_detector import detector
from tools.utils.evaluation_recall import BasicSceneGraphEvaluator

from dataloader.AG.action_genome import AG, cuda_collate_fn

from lib.tempura import TEMPURA                               

print('\n>>>>>>>>>>>>> AG TEMPURA Train.py <<<<<<<<<<<<<')
conf = Config()
seed = set_seed_and_igwarn()
gpu_device = torch.device('cuda:0')
log, log_val, model_save_path, tf_path = set_train_dir(conf.save_path, conf.mode)
print('The CKPT saved here:', conf.save_path)
print('The Tensorboard saved here:', tf_path)

print('\n>>>>>>>>>>>>>>>> Dataset Load <<<<<<<<<<<<<<<<')
AG_dataset_train = AG(mode="train", datasize=conf.datasize, data_path=conf.data_path, filter_nonperson_box_frame=True,
                      filter_small_box=False if conf.mode == 'predcls' else True)
dataloader_train = torch.utils.data.DataLoader(AG_dataset_train, shuffle=True, num_workers=4,
                                               collate_fn=cuda_collate_fn, pin_memory=False, generator=torch.Generator().manual_seed(seed))

AG_dataset_test = AG(mode="test", datasize=conf.datasize, data_path=conf.data_path, filter_nonperson_box_frame=True,
                     filter_small_box=False if conf.mode == 'predcls' else True)
dataloader_test = torch.utils.data.DataLoader(AG_dataset_test, shuffle=False, num_workers=4,
                                              collate_fn=cuda_collate_fn, pin_memory=False, generator=torch.Generator().manual_seed(seed))

print('\n>>>>>>>>>>>>> Object Detector Load <<<<<<<<<<<<<')
object_detector = detector(train=True, object_classes=AG_dataset_train.object_classes, use_SUPPLY=True, mode=conf.mode).to(device=gpu_device)
object_detector.eval()
print('FasterRCNN eval mode, Freeze weight, Just use pretrained ckpt.')

print('\n>>>>>>>>>>>>>>>>> TEMPURA Load <<<<<<<<<<<<<<<<<')
model =  TEMPURA(mode=conf.mode,
               attention_class_num=len(AG_dataset_train.attention_relationships),
               spatial_class_num=len(AG_dataset_train.spatial_relationships),
               contact_class_num=len(AG_dataset_train.contacting_relationships),
               obj_classes=AG_dataset_train.object_classes,
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
               tracking= conf.tracking).to(device=gpu_device)

# load_ckpt
if False:
    load_path = 'checkpoint/AG_clip_conloss_temp_cons_str_sem/predcls/models/best_Mrecall_model.tar'
    ckpt = torch.load(load_path, map_location=gpu_device)
    model.load_state_dict(ckpt['state_dict'], strict=True)
    print('CKPT {} is loaded'.format(load_path))

print('\n>>>>>>>>>>>> Evaluator Definition <<<<<<<<<<<<')
evaluator =BasicSceneGraphEvaluator(mode=conf.mode,
                                    AG_object_classes=AG_dataset_train.object_classes,
                                    AG_all_predicates=AG_dataset_train.relationship_classes,
                                    AG_attention_predicates=AG_dataset_train.attention_relationships,
                                    AG_spatial_predicates=AG_dataset_train.spatial_relationships,
                                    AG_contacting_predicates=AG_dataset_train.contacting_relationships,
                                    iou_threshold=0.5,
                                    constraint='with')
print(f'Train Evaluator: With Constraint Setting')

# Loss function, default Multi-label margin loss ====================== #
weights = torch.ones(len(model.obj_classes))
weights[0] = conf.eos_coef

ce_loss_obj = nn.CrossEntropyLoss(weight=weights.to(device=gpu_device),reduction='none')
ce_loss_rel = nn.CrossEntropyLoss(reduction='none')
bce_loss = nn.BCELoss(reduction='none')
con_loss = metric_loss.ContrastiveLoss(pos_margin=0, neg_margin=1)

# Optimizer =========================================================== #
for name, value in model.named_parameters():
    if 'object_classifier' in name and conf.mode == 'predcls':
        value.requires_grad = False
     
learned_params = model.parameters()
optimizer = AdamW(learned_params, lr=conf.lr,  betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1)

scheduler = ExponentialLR(optimizer, gamma=0.8)
warmup_scheduler = warmup.ExponentialWarmup(optimizer, warmup_period=conf.warmup)
print(f'Optimizer: {conf.optimizer},  Scheduler: ExponentialLR')

# Metrics Parameter =================================================== #
tr, best_recall, best_Mrecall, loss_step = [], 0, 0, 0
writer = SummaryWriter(tf_path)
layout = {
    "Train_Wrtier": {
        "Loss": ["Multiline", ["att_loss", "spatial_loss", "contact_loss", "total_loss"]],
        "Recall@K": ["Multiline", ["Recall@10", "Recall@20", "Recall@50", "Recall@100"]],
        "MRecall@K": ["Multiline", ["MRecall@10", "MRecall@20", "MRecall@50", "MRecall@100"]],
        "lr": "lr"
    }}

writer.add_custom_scalars(layout)
print('loss: obj_loss + att_loss + spatial_loss + contact_loss = total_loss')
print('Metric: Recall@K, MRecall@K')

print('\n>>>>>>>>>>>>>>> Model Training <<<<<<<<<<<<<<<')
video_graph_memory = {}
for epoch in range(conf.nepoch):
        
    unc_vals = uncertainty_values(obj_classes=len(model.obj_classes),
                                    attention_class_num=model.attention_class_num,
                                    spatial_class_num=model.spatial_class_num,
                                    contact_class_num=model.contact_class_num)
    model.train()
    object_detector.is_train = True
    
    start = time.time()
    train_iter = iter(dataloader_train)
    test_iter = iter(dataloader_test)
    len_train_batch = len(dataloader_train)
    
    # Train
    writer.add_scalar("lr", scheduler.get_lr()[-1], epoch)
    for b in tqdm(range(len_train_batch)):
     
        data = next(train_iter)
        im_data = copy.deepcopy(data[0].to(device=gpu_device))
        im_info = copy.deepcopy(data[1].to(device=gpu_device))
        gt_boxes = copy.deepcopy(data[2].to(device=gpu_device))
        num_boxes = copy.deepcopy(data[3].to(device=gpu_device))
        gt_annotation = AG_dataset_train.gt_annotations[data[4]]
                
        # prevent gradients to FasterRCNN
        with torch.no_grad():
            entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation ,im_all=None)
            
        if conf.tracking:
            get_sequence(entry, gt_annotation, (im_info[0][:2]/im_info[0,2]).cpu().data,conf.mode)            
        
        pred = model(entry, phase='train', unc=False)        
        
        uncertainty_computation(data,AG_dataset_train,
                                object_detector,model,unc_vals,gpu_device,
                                conf.save_path,
                                obj_unc=conf.obj_unc,obj_mem=conf.obj_mem_compute,
                                background_mem=False,rel_unc=conf.rel_unc,
                                tracking=conf.tracking)
            
        attention_distribution = pred["attention_distribution"]
        spatial_distribution = pred["spatial_distribution"]
        contact_distribution = pred["contacting_distribution"]        
        
        # 예측 결과 shape 맞추기.
        # bce loss
        attention_label = torch.tensor(pred["attention_gt"], dtype=torch.long).to(device=attention_distribution.device).squeeze()
        spatial_label = torch.zeros([len(pred["spatial_gt"]), 6], dtype=torch.float32).to(device=attention_distribution.device)
        contact_label = torch.zeros([len(pred["contacting_gt"]), 17], dtype=torch.float32).to(device=attention_distribution.device)
        
        for i in range(len(pred["spatial_gt"])):
            spatial_label[i, pred["spatial_gt"][i]] = 1
            contact_label[i, pred["contacting_gt"][i]] = 1

        # 각 술어 카테고리별 loss 계산.
        losses = {}
        if conf.mode == 'sgcls' or conf.mode == 'sgdet':
            losses['object_loss'] = ce_loss_obj(pred['distribution'], pred['labels'])
            loss_weighting = conf.obj_loss_weighting
    
            losses['object_loss'] = losses['object_loss'].mean()
            
            if conf.obj_con_loss:
                losses['object_contrastive_loss'] = conf.lambda_con*con_loss(pred['object_mem_features'], pred['labels'])

        losses["attention_relation_loss"] = ce_loss_rel(attention_distribution, attention_label)
        losses["spatial_relation_loss"] = bce_loss(spatial_distribution, spatial_label) #bce_loss
        losses["contacting_relation_loss"] = bce_loss(contact_distribution, contact_label) #bce_loss
        
        # Loss Weighting & Average
        for rel in ['attention','spatial','contacting']: 
            losses[rel+'_relation_loss'] = losses[rel+'_relation_loss'].mean()
        
        # Relationship Contrastive Loss
        if conf.use_ctl_loss:            
            # losses['attention_con_loss'] = 0.25*con_loss(attention_distribution, attention_label).mean()
            losses['spatial_con_loss'] = 0.2*con_loss(spatial_distribution, torch.argmax(spatial_label, dim=1)).mean()
            losses['contact_con_loss'] = 0.2*con_loss(contact_distribution, torch.argmax(contact_label, dim=1)).mean()
        
        # Temporal Consistency Loss
        if conf.use_cons_str_loss:  
            losses['structure_temp_loss'] = pred['structure_temp_loss'].mean()*2500 # 2500
        if conf.use_cons_sem_loss:  
            losses['semantic_temp_loss'] = pred['semantic_temp_loss'].mean()*2500 # 2500
        
        optimizer.zero_grad()
        loss = sum(losses.values())
       
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
        optimizer.step()
        
        losses['total_loss'] = loss
        tr.append(pd.Series({x: y.item() for x, y in losses.items()}))
        
        # Logger & Tensorboard report: 비디오 100개 마다 로그 기록.
        log_iter = conf.log_iter
        if (b+1) % log_iter == 0 and (b+1) >= log_iter:
            time_per_batch = (time.time() - start) / log_iter
            str_print = "\nepoch:{:2d}  batch:{:5d}/{:5d}  {:.3f}s/batch, {:.1f}m/epoch".format(epoch, b, len(dataloader_train),
                                                                                time_per_batch, len(dataloader_train) * time_per_batch / 60)
            print(str_print, flush=True)

            log.write(str_print+'\n')

            mn = pd.concat(tr[-log_iter:], axis=1).mean(1)
            print(mn, flush=True)
            for k in list(mn.keys()):
                str_print = '{} : {:5f}'.format(k, mn[k])
                log.write(str_print+'\n')
            
            start = time.time()
            
            # Tensorboard Report
            writer.add_scalar("att_loss", mn['attention_relation_loss'].round(6), loss_step)
            writer.add_scalar("spatial_loss", mn['spatial_relation_loss'].round(6), loss_step)
            writer.add_scalar("contact_loss", mn['contacting_relation_loss'].round(6), loss_step)
            writer.add_scalar("total_loss", mn["total_loss"].round(6), loss_step)
            
            if conf.use_ctl_loss:
                writer.add_scalar("spatial_con_loss", mn['spatial_con_loss'].round(6), loss_step)
                writer.add_scalar("contact_con_loss", mn['contact_con_loss'].round(6), loss_step)
            
            if conf.use_cons_str_loss:
                writer.add_scalar("structure_temp_loss", mn['structure_temp_loss'].round(6), loss_step)                
            if conf.use_cons_sem_loss:
                writer.add_scalar("semantic_temp_loss", mn['semantic_temp_loss'].round(6), loss_step)   
            
            loss_step += 1
    
    # Validation
    model.eval()
    object_detector.is_train = False
    with torch.no_grad():
        for b in tqdm(range(len(dataloader_test))):
            
            data = next(test_iter)

            im_data = copy.deepcopy(data[0].cuda(0))
            im_info = copy.deepcopy(data[1].cuda(0))
            gt_boxes = copy.deepcopy(data[2].cuda(0))
            num_boxes = copy.deepcopy(data[3].cuda(0))
            gt_annotation = AG_dataset_test.gt_annotations[data[4]]       
                        
            entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)
            
            if conf.tracking:
                get_sequence(entry, gt_annotation, (im_info[0][:2]/im_info[0,2]).cpu().data,conf.mode)

            pred = model(entry, phase='test', unc=False)
            
            evaluator.evaluate_scene_graph(gt_annotation, pred)
        print('-----------'*3, flush=True)
    
    recall = np.mean(evaluator.result_dict[conf.mode + "_recall"][20])
    mrecall = evaluator.calc_mrecall()[20]
    
    # Log & Tensorboard Writing
    log_val.write('epoch {} validation results:'.format(epoch)+'\n')
    evaluator.print_stats(log_file=log_val, log_writer=writer, log_epoch=epoch)
        
    if epoch % 5 == 0:
        if epoch > 0:
            if len(model.object_classifier.obj_memory) == 0:
                object_memory = []
            else:
                object_memory = model.object_classifier.obj_memory.to('cpu')
            rel_memory = model.rel_memory
            if len(rel_memory) != 0:
                rel_memory = {k:rel_memory[k].to('cpu') for k in rel_memory.keys()}
        else:
            object_memory = []
            rel_memory = []
        torch.save({"state_dict": model.state_dict(),
                    'object_memory':object_memory,
                    'rel_memory':rel_memory}, os.path.join(model_save_path, f"checkpoint_{epoch}_model.tar".format(epoch)))
    
    # Recall 최고 기록 갱신
    if recall > best_recall:
        best_recall = recall
        str_print = 'new best recall of {} at epoch {}'.format(best_recall,epoch)
        if epoch > 0:
            if len(model.object_classifier.obj_memory) == 0:
                object_memory = []
            else:
                object_memory = model.object_classifier.obj_memory.to('cpu')
            rel_memory = model.rel_memory
            if len(rel_memory) != 0:
                rel_memory = {k:rel_memory[k].to('cpu') for k in rel_memory.keys()}
        else:
            object_memory = []
            rel_memory = []
        
        print(str_print+'\n', flush=True)
        log_val.write(str_print+'\n')    
        torch.save({"state_dict": model.state_dict(),
                    'object_memory':object_memory,
                    'rel_memory':rel_memory}, os.path.join(model_save_path, "best_recall_model.tar".format(epoch)))
    
    # MRecall 최고 기록 갱신
    if mrecall > best_Mrecall:
        best_Mrecall = mrecall
        str_print = 'new best Mrecall of {} at epoch {}'.format(best_Mrecall,epoch)
        print(str_print+'\n', flush=True)
        log_val.write(str_print+'\n')
        if epoch > 0:
            object_memory = model.object_classifier.obj_memory.to('cpu')
            rel_memory = model.rel_memory
            rel_memory = {k:rel_memory[k].to('cpu') for k in rel_memory.keys()}
        else:
            object_memory = []
            rel_memory = []
        torch.save({"state_dict": model.state_dict(),
                    'object_memory':object_memory,
                    'rel_memory':rel_memory}, os.path.join(model_save_path, "best_Mrecall_model.tar".format(epoch)))
    
    evaluator.reset_result()
    
    # Scheduler Step
    past_lr = scheduler.get_lr()[-1]
    with warmup_scheduler.dampening():
        scheduler.step()
    current_lr = scheduler.get_lr()[-1]
    print(f'Epoch {epoch} lr updated: {past_lr} --> {current_lr}')
    
    # Memory Computation
    print('computing memory \n', flush=True)
    rel_class_num = {'attention':model.attention_class_num,
                        'spatial': model.spatial_class_num,
                        'contacting': model.contact_class_num}
    if conf.tracking:
        obj_feature_dim = 2048+200+128
    else:
        obj_feature_dim = 1024
    rel_memory, obj_memory = memory_computation(unc_vals,
                conf.save_path,rel_class_num,
                   len(model.obj_classes),obj_feature_dim=obj_feature_dim,
                   rel_feature_dim=1936, obj_weight_type=conf.obj_mem_weight_type, 
                   
                   rel_weight_type=conf.rel_mem_weight_type,
                   obj_mem=conf.obj_mem_compute,obj_unc=conf.obj_unc,
                    include_bg_mem = False)
        
    model.object_classifier.obj_memory = obj_memory.to(gpu_device)
    model.rel_memory = {k:rel_memory[k].to(gpu_device) for k in rel_memory.keys()}

log.close()
log_val.close()
writer.close()
print('\n>>>>>>>> TEMPURA Train.py Complete <<<<<<<<')