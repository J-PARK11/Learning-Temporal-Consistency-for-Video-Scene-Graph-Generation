import torch
import torch.nn as nn
import numpy as np
import pickle
from functools import reduce
from tools.utils.pytorch_misc import intersect_2d, argsort_desc
from tools.utils.fpn.box_intersections_cpu.bbox import bbox_overlaps

class BasicSceneGraphEvaluator:
    def __init__(self, mode, AG_object_classes, AG_all_predicates,
                 AG_attention_predicates, AG_spatial_predicates, AG_contacting_predicates,
                 iou_threshold=0.5,  constraint=False, semithreshold=None, output_dir='output/'):
        
        # Object & Predicate Classes
        self.AG_object_classes = AG_object_classes
        self.AG_all_predicates = AG_all_predicates
        self.AG_attention_predicates = AG_attention_predicates
        self.AG_spatial_predicates = AG_spatial_predicates
        self.AG_contacting_predicates = AG_contacting_predicates
        
        # Basic Argument
        self.result_dict, self.per_class_recall = {}, {}    # per_class_recall: M_Recall per Predicate Ratio 
        self.mode, self.constraint = mode, constraint       # constraint == True if Semi Constraint Active
        self.result_dict[self.mode + '_recall'] = {10: [], 20: [], 50: [], 100: []}
        self.iou_threshold, self.semithreshold = iou_threshold, semithreshold
        self.output_dir, self.tot_all_predicates = output_dir, len(AG_all_predicates)
        self.gt_obj_list, self.pred_obj_list = [], []

    # Reset the 'Result_dictionary'
    def reset_result(self):
        self.result_dict[self.mode + '_recall'] = {10: [], 20: [], 50: [], 100: []}
    
    # Calculate the Mean of Recall 
    def calc_mrecall(self):
        
        # K별 recall 평균 계산.
        for k, v in self.result_dict[self.mode + '_recall'].items():
            avg = 0
            self.per_class_recall[k] = {}
            # 술어 클래스별 recall 평균 계산
            for idx in range(self.tot_all_predicates):
                # Recall_hit / Recall_Count
                tmp_avg = float(self.result_dict[self.mode + '_recall_hit'][k][idx]) / float(self.result_dict[self.mode +'_recall_count'] [k][idx] + 1e-10)
                avg += tmp_avg
                self.per_class_recall[k][self.AG_all_predicates[idx]]= tmp_avg
            # Mean_recall dict가 없으면 생성.
            if (self.mode + '_Mrecall') not in self.result_dict:
                self.result_dict[self.mode + '_Mrecall'] = {}
            self.result_dict[self.mode + '_Mrecall'][k] = avg/self.tot_all_predicates

        return self.result_dict[self.mode + '_Mrecall']
    
    # Print Log & Mean Recall
    def print_stats(self, log_file=None, log_writer=None, log_epoch=None, metric=None):
        print(f'--------- {metric}_{self.mode} ---------')
        if log_file:
            log_file.write('-'*15+self.constraint+'_constraint'+'\n')
        
        for k, v in self.result_dict[self.mode + '_recall'].items():
            print('R@%i: %f' % (k, np.mean(v)), flush=True)
            if log_file:
                log_file.write('R@%i: %f' % (k, np.mean(v)) + ' \n')
            
            avg = 0
            per_class_recall = {}
            for idx in range(self.tot_all_predicates):
                 tmp_avg = float(self.result_dict[self.mode + '_recall_hit'][k][idx]) / float(self.result_dict[self.mode +'_recall_count'] [k][idx] + 1e-10)
                 avg += tmp_avg
                 per_class_recall[self.AG_all_predicates[idx]]= tmp_avg

            print('mR@%i: %f'% (k, avg/self.tot_all_predicates), flush=True)
            if log_file:
                log_file.write('mR@%i: %f'% (k, avg/self.tot_all_predicates) + ' \n')
            
            if log_writer:
                log_writer.add_scalar(f"{metric}_R@K/{metric}_R@{k}", np.mean(v), log_epoch)
                log_writer.add_scalar(f"{metric}_MR@K/{metric}_MR@{k}", avg/self.tot_all_predicates, log_epoch)
            
            if self.output_dir:
                with open(self.output_dir+
                          self.mode+'_'+
                          self.constraint+'_constraint_per_cls_recall_at_{}.pkl'.format(k),'wb') as f:
                          pickle.dump(per_class_recall,f)
        
    # Evaluate SGG Prediction Result via GT for one batch        
    def evaluate_scene_graph(self, gt, pred):
        counter, video_pred_dict = 0, {}
        for idx, frame_gt in enumerate(gt):
            # Now there is no person box! we assume that person box index == 0
            gt_boxes = np.zeros([len(frame_gt), 4])
            gt_classes = np.zeros(len(frame_gt))
            gt_relations, human_idx = [], 0
            
            gt_classes[human_idx] = 1
            gt_boxes[human_idx] = frame_gt[0]['person_bbox']
            frame_id = frame_gt[0]['frame']
            
            # BBox & Class per Frame observed Object
            for m, n in enumerate(frame_gt[1:]):
                # Each Pair of Object {bbox, class}
                gt_boxes[m+1,:] = n['bbox']
                gt_classes[m+1] = n['class']
            
                #spatial and contacting relationship could be multiple
                gt_relations.append([human_idx, m+1, self.AG_all_predicates.index(self.AG_attention_predicates[n['attention_relationship']])]) # for attention triplet <human-object-predicate>_
                for spatial in n['spatial_relationship'].numpy().tolist():
                    gt_relations.append([m+1, human_idx, self.AG_all_predicates.index(self.AG_spatial_predicates[spatial])]) # for spatial triplet <object-human-predicate>
                for contact in n['contacting_relationship'].numpy().tolist():
                    gt_relations.append([human_idx, m+1, self.AG_all_predicates.index(self.AG_contacting_predicates[contact])])  # for contact triplet <human-object-predicate>

            gt_entry = {
                'gt_classes': gt_classes,               # [len(frame_gt), 1]
                'gt_relations': np.array(gt_relations), # [len(frame_gt), 3]
                'gt_boxes': gt_boxes,                   # [len(frame_gt), 4]
            }
            
            # Constraint & Mode categorized GT_obj_list appending
            if self.constraint == 'no' and self.mode != 'predcls':
                self.gt_obj_list.append({
                    'boxes':torch.tensor(gt_boxes),
                    'labels':torch.tensor( gt_classes),
                })
                
            # Prediction Data Rearranging
            rels_i = np.concatenate((pred['pair_idx'][pred['im_idx'] == idx].cpu().clone().numpy(),             #attention
                                     pred['pair_idx'][pred['im_idx'] == idx].cpu().clone().numpy()[:,::-1],     #spatial
                                     pred['pair_idx'][pred['im_idx'] == idx].cpu().clone().numpy()), axis=0)    #contacting


            pred_scores_1 = np.concatenate((pred['attention_distribution'][pred['im_idx'] == idx].cpu().numpy(),
                                            np.zeros([pred['pair_idx'][pred['im_idx'] == idx].shape[0], pred['spatial_distribution'].shape[1]]),
                                            np.zeros([pred['pair_idx'][pred['im_idx'] == idx].shape[0], pred['contacting_distribution'].shape[1]])), axis=1)
            pred_scores_2 = np.concatenate((np.zeros([pred['pair_idx'][pred['im_idx'] == idx].shape[0], pred['attention_distribution'].shape[1]]),
                                            pred['spatial_distribution'][pred['im_idx'] == idx].cpu().numpy(),
                                            np.zeros([pred['pair_idx'][pred['im_idx'] == idx].shape[0], pred['contacting_distribution'].shape[1]])), axis=1)
            pred_scores_3 = np.concatenate((np.zeros([pred['pair_idx'][pred['im_idx'] == idx].shape[0], pred['attention_distribution'].shape[1]]),
                                            np.zeros([pred['pair_idx'][pred['im_idx'] == idx].shape[0], pred['spatial_distribution'].shape[1]]),
                                            pred['contacting_distribution'][pred['im_idx'] == idx].cpu().numpy()), axis=1)

            # Metric per 'Pred_entry' Dictionary Definition: Predcls의 경우만 GT Object label을 사용함.
            if self.mode == 'predcls':
                pred_entry = {
                    'pred_boxes': pred['boxes'][:,1:].cpu().clone().numpy(),
                    'pred_classes': pred['labels'].cpu().clone().numpy(),
                    'pred_rel_inds': rels_i,
                    'obj_scores': pred['scores'].cpu().clone().numpy(),
                    'rel_scores': np.concatenate((pred_scores_1, pred_scores_2, pred_scores_3), axis=0)
                }
            else:       # ['sgcls', 'phrdets', 'sggen']
                pred_entry = {
                    'pred_boxes': pred['boxes'][:, 1:].cpu().clone().numpy(),
                    'pred_classes': pred['pred_labels'].cpu().clone().numpy(),
                    'pred_rel_inds': rels_i,
                    'obj_scores': pred['pred_scores'].cpu().clone().numpy(),
                    'rel_scores': np.concatenate((pred_scores_1, pred_scores_2, pred_scores_3), axis=0)
                }
            
            # Constraint & Mode categorized Pred_obj_list appending
            if self.constraint == 'no' and self.mode != 'predcls':             
                self.pred_obj_list.append({
                        'boxes':pred['boxes'][counter:counter+len(frame_gt), 1:].cpu().clone(),
                        'scores':pred['pred_scores'][counter:counter+len(frame_gt)].cpu().clone(),
                        'labels': pred['pred_labels'][counter:counter+len(frame_gt)].cpu().clone(),
                    })         
                counter = counter + len(frame_gt)
            
            # Evaluate from dict
            _, _,_, rel_scores, pred_triplets,pred_triplet_boxes = evaluate_from_dict(gt_entry, pred_entry, self.mode, 
                                                                                    self.result_dict,iou_thresh=self.iou_threshold, 
                                                                                    method=self.constraint, threshold=self.semithreshold, 
                                                                                    tot_all_predicates = self.tot_all_predicates)
            
            # video_pred_dict[frame_id] = {'triplet_scores':rel_scores,
            #                              'triplet_labels':pred_triplets,
            #                              'triplet_boxes':pred_triplet_boxes}
            
           
        # return video_pred_dict
            
def evaluate_from_dict(gt_entry, pred_entry, mode, result_dict, method=None, threshold = 0.9,tot_all_predicates=26, **kwargs):
    """
    Shortcut to doing evaluate_recall from dict
    :param gt_entry: Dictionary containing gt_relations, gt_boxes, gt_classes
    :param pred_entry: Dictionary containing pred_rels, pred_boxes (if detection), pred_classes
    :param result_dict:
    :param kwargs:
    :return:
    """
    
    # Argument Definition
    gt_rels = gt_entry['gt_relations']
    gt_boxes = gt_entry['gt_boxes'].astype(float)
    gt_classes = gt_entry['gt_classes']
    
    pred_rel_inds = pred_entry['pred_rel_inds']
    rel_scores = pred_entry['rel_scores']

    pred_boxes = pred_entry['pred_boxes'].astype(float)
    pred_classes = pred_entry['pred_classes']
    obj_scores = pred_entry['obj_scores']

    # Semi Constraint Evaluation
    if method == 'semi':
        pred_rels = []
        predicate_scores = []
        for i, j in enumerate(pred_rel_inds):
            
            # Attention Predicate
            if rel_scores[i,0]+rel_scores[i,1] > 0:
                pred_rels.append(np.append(j,rel_scores[i].argmax()))
                predicate_scores.append(rel_scores[i].max())
            
            # Spatial Predicate
            elif rel_scores[i,3]+rel_scores[i,4] > 0:
                for k in np.where(rel_scores[i]>threshold)[0]:
                    pred_rels.append(np.append(j, k))
                    predicate_scores.append(rel_scores[i,k])
                    
            # Contact Predicate
            elif rel_scores[i,9]+rel_scores[i,10] > 0:
                for k in np.where(rel_scores[i]>threshold)[0]:
                    pred_rels.append(np.append(j, k))
                    predicate_scores.append(rel_scores[i,k])
        pred_rels = np.array(pred_rels)
        predicate_scores = np.array(predicate_scores)
    
    # No Constraint Evaluation
    elif method == 'no':
        obj_scores_per_rel = obj_scores[pred_rel_inds].prod(1)
        overall_scores = obj_scores_per_rel[:, None] * rel_scores
        score_inds = argsort_desc(overall_scores)[:100]
        pred_rels = np.column_stack((pred_rel_inds[score_inds[:, 0]], score_inds[:, 1]))
        predicate_scores = rel_scores[score_inds[:, 0], score_inds[:, 1]]

    # With Constraint Evaluation
    else:
        pred_rels = np.column_stack((pred_rel_inds, rel_scores.argmax(1))) #1+  dont add 1 because no dummy 'no relations'
        predicate_scores = rel_scores.max(1)

    pred_to_gt, pred_5ples, rel_scores, orig_rel_scores, orig_pred_triplets, orig_pred_triplet_boxes = evaluate_recall(
                gt_rels, gt_boxes, gt_classes,
                pred_rels, pred_boxes, pred_classes,
                predicate_scores, obj_scores, phrdet= mode=='phrdet',
                **kwargs)

    for k in result_dict[mode + '_recall']:
        match = reduce(np.union1d, pred_to_gt[:k])

        for idx in range(len(match)):
            local_label = gt_rels[int(match[idx]),2]
            
            # Initialize
            if (mode + '_recall_hit') not in result_dict:
                result_dict[mode + '_recall_hit'] = {}
                
            # K별 m_recall_hit 기입
            if k not in result_dict[mode + '_recall_hit']:
                result_dict[mode + '_recall_hit'][k] = [0] * (tot_all_predicates)
            result_dict[mode + '_recall_hit'][k][int(local_label)] += 1

        for idx in range(gt_rels.shape[0]):
            local_label = gt_rels[idx,2]
            
            # Initialize
            if (mode + '_recall_count') not in result_dict:
                result_dict[mode + '_recall_count'] = {}
            
            # K별 m_recall_count 기입
            if k not in result_dict[mode + '_recall_count']:
                result_dict[mode + '_recall_count'][k] = [0] * (tot_all_predicates)
            result_dict[mode + '_recall_count'][k][int(local_label)] += 1

        rec_i = float(len(match)) / float(gt_rels.shape[0])
        result_dict[mode + '_recall'][k].append(rec_i)
        
    return pred_to_gt, pred_5ples, rel_scores, orig_rel_scores, orig_pred_triplets, orig_pred_triplet_boxes

# ================== Interior Submodule ================== #

def evaluate_recall(gt_rels, gt_boxes, gt_classes,
                    pred_rels, pred_boxes, pred_classes, rel_scores=None, cls_scores=None,
                    iou_thresh=0.5, phrdet=False):
    """
    Evaluates the recall
    :param gt_rels: [#gt_rel, 3] array of GT relations
    :param gt_boxes: [#gt_box, 4] array of GT boxes
    :param gt_classes: [#gt_box] array of GT classes
    :param pred_rels: [#pred_rel, 3] array of pred rels. Assumed these are in sorted order
                      and refer to IDs in pred classes / pred boxes
                      (id0, id1, rel)
    :param pred_boxes:  [#pred_box, 4] array of pred boxes
    :param pred_classes: [#pred_box] array of predicted classes for these boxes
    :return: pred_to_gt: Matching from predicate to GT
             pred_5ples: the predicted (id0, id1, cls0, cls1, rel)
             rel_scores: [cls_0score, cls1_score, relscore]
                   """
    
    # Except case: Empty Prediction Relationship           
    if pred_rels.size == 0:
        return [[]], np.zeros((0,5)), np.zeros(0), [],[],[]

    num_gt_boxes = gt_boxes.shape[0]
    num_gt_relations = gt_rels.shape[0]
    num_boxes = pred_boxes.shape[0]
    assert num_gt_relations != 0
    assert pred_rels[:,:2].max() < pred_classes.shape[0]

    # Make GT Triplet
    gt_triplets, gt_triplet_boxes, _ = _triplet(gt_rels[:, 2],
                                                gt_rels[:, :2],
                                                gt_classes,
                                                gt_boxes)

    # Make Prediction Triplet
    pred_triplets, pred_triplet_boxes, relation_scores = _triplet(pred_rels[:,2],
                                                                  pred_rels[:,:2],
                                                                  pred_classes,
                                                                  pred_boxes, 
                                                                  rel_scores,
                                                                  cls_scores)
    # Original Pred Triplet 
    orig_pred_triplets = pred_triplets
    orig_pred_triplet_boxes = pred_triplet_boxes
    orig_relation_scores = relation_scores
    
    sorted_scores = relation_scores.prod(1)
    pred_triplets = pred_triplets[sorted_scores.argsort()[::-1],:]
    pred_triplet_boxes = pred_triplet_boxes[sorted_scores.argsort()[::-1],:]
    relation_scores = relation_scores[sorted_scores.argsort()[::-1],:]
    scores_overall = relation_scores.prod(1)

    # Sorted Integrity Check
    if not np.all(scores_overall[1:] <= scores_overall[:-1] + 1e-5):
        print("Somehow the relations weren't sorted properly: \n{}".format(scores_overall))

    # Compute recall. It's most efficient to match once and then do recall after
    pred_to_gt = _compute_pred_matches(
        gt_triplets,
        pred_triplets,
        gt_triplet_boxes,
        pred_triplet_boxes,
        iou_thresh,
        phrdet=phrdet,
    )
    
    # Contains some extra stuff for visualization. Not needed.
    pred_5ples = np.column_stack((pred_rels[:,:2],
                                  pred_triplets[:, [0, 2, 1]]))

    return pred_to_gt, pred_5ples, relation_scores, orig_relation_scores, orig_pred_triplets, orig_pred_triplet_boxes


def _triplet(predicates, relations, classes, boxes,
             predicate_scores=None, class_scores=None):
    """
    format predictions into triplets
    :param predicates: A 1d numpy array of num_boxes*(num_boxes-ĺeftright) predicates, corresponding to
                       each pair of possibilities
    :param relations: A (num_boxes*(num_boxes-ĺeftright), 2.0) array, where each row represents the boxes
                      in that relation
    :param classes: A (num_boxes) array of the classes for each thing.
    :param boxes: A (num_boxes,4) array of the bounding boxes for everything.
    :param predicate_scores: A (num_boxes*(num_boxes-ĺeftright)) array of the scores for each predicate
    :param class_scores: A (num_boxes) array of the likelihood for each object.
    :return: Triplets: (num_relations, 3) array of class, relation, class
             Triplet boxes: (num_relation, 8) array of boxes for the parts
             Triplet scores: num_relation array of the scores overall for the triplets
    """
    assert (predicates.shape[0] == relations.shape[0])

    sub_ob_classes = classes[relations[:, :2]]
    triplets = np.column_stack((sub_ob_classes[:, 0], predicates, sub_ob_classes[:, 1]))
    triplet_boxes = np.column_stack((boxes[relations[:, 0]], boxes[relations[:, 1]]))

    triplet_scores = None
    if predicate_scores is not None and class_scores is not None:
        triplet_scores = np.column_stack((
            class_scores[relations[:, 0]],
            class_scores[relations[:, 1]],
            predicate_scores,
        ))

    return triplets, triplet_boxes, triplet_scores

def _compute_pred_matches(gt_triplets, pred_triplets,
                 gt_boxes, pred_boxes, iou_thresh, phrdet=False):
    """
    Given a set of predicted triplets, return the list of matching GT's for each of the
    given predictions
    :param gt_triplets:
    :param pred_triplets:
    :param gt_boxes:
    :param pred_boxes:
    :param iou_thresh:
    :return:
    """
    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    # The rows correspond to GT triplets, columns to pred triplets
    
    keeps = intersect_2d(gt_triplets, pred_triplets)
    gt_has_match = keeps.any(1)
    pred_to_gt = [[] for x in range(pred_boxes.shape[0])]
    for gt_ind, gt_box, keep_inds in zip(np.where(gt_has_match)[0],
                                         gt_boxes[gt_has_match],
                                         keeps[gt_has_match],
                                         ):
        boxes = pred_boxes[keep_inds]
        if phrdet:
            # Evaluate where the union box > 0.5
            gt_box_union = gt_box.reshape((2, 4))
            gt_box_union = np.concatenate((gt_box_union.min(0)[:2], gt_box_union.max(0)[2:]), 0)

            box_union = boxes.reshape((-1, 2, 4))
            box_union = np.concatenate((box_union.min(1)[:,:2], box_union.max(1)[:,2:]), 1)

            inds = bbox_overlaps(gt_box_union[None], box_union)[0] >= iou_thresh

        else:
            sub_iou = bbox_overlaps(gt_box[None,:4], boxes[:, :4])[0]
            obj_iou = bbox_overlaps(gt_box[None,4:], boxes[:, 4:])[0]

            inds = (sub_iou >= iou_thresh) & (obj_iou >= iou_thresh)

        for i in np.where(keep_inds)[0][inds]:
            pred_to_gt[i].append(int(gt_ind))
            
    return pred_to_gt

def Get_AG_Evaluator(mode, dataset, output_path):
    
    evaluator1 = BasicSceneGraphEvaluator(
        mode=mode,
        AG_object_classes=dataset.object_classes,
        AG_all_predicates=dataset.relationship_classes,
        AG_attention_predicates=dataset.attention_relationships,
        AG_spatial_predicates=dataset.spatial_relationships,
        AG_contacting_predicates=dataset.contacting_relationships,
        output_dir = output_path,
        iou_threshold=0.5,
        constraint='with')

    evaluator2 = BasicSceneGraphEvaluator(
        mode=mode,
        AG_object_classes=dataset.object_classes,
        AG_all_predicates=dataset.relationship_classes,
        AG_attention_predicates=dataset.attention_relationships,
        AG_spatial_predicates=dataset.spatial_relationships,
        AG_contacting_predicates=dataset.contacting_relationships,
        output_dir = output_path,
        iou_threshold=0.5,
        constraint='semi', semithreshold=0.9)

    evaluator3 = BasicSceneGraphEvaluator(
        mode=mode,
        AG_object_classes=dataset.object_classes,
        AG_all_predicates=dataset.relationship_classes,
        AG_attention_predicates=dataset.attention_relationships,
        AG_spatial_predicates=dataset.spatial_relationships,
        AG_contacting_predicates=dataset.contacting_relationships,
        output_dir = output_path,
        iou_threshold=0.5,
        constraint='no')
    
    return evaluator1, evaluator2, evaluator3