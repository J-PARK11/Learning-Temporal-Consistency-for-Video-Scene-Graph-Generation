import os
import cv2
import torch
import pickle
import random
import numpy as np
from PIL import Image
from cv2 import imread
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize

class KCC_AG(Dataset):
    
    def __init__(self, data_path='/data/AG/'):

        AG_root_path = data_path
        self.data_root_path = 'kcc_demo/dataset/'
        self.video_list = os.listdir(self.data_root_path)

        # Collect the object classes
        self.object_classes = ['__background__']
        with open(os.path.join(AG_root_path, 'annotations/object_classes.txt'), 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                self.object_classes.append(line)
        f.close()
        
        self.object_classes[9] = 'closet/cabinet'
        self.object_classes[11] = 'cup/glass/bottle'
        self.object_classes[23] = 'paper/notebook'
        self.object_classes[24] = 'phone/camera'
        self.object_classes[31] = 'sofa/couch'

        # Collect & Relabel relationship classes
        self.relationship_classes = []
        with open(os.path.join(AG_root_path, 'annotations/relationship_classes.txt'), 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                self.relationship_classes.append(line)
        f.close()
        
        self.relationship_classes[0] = 'looking_at'
        self.relationship_classes[1] = 'not_looking_at'
        self.relationship_classes[5] = 'in_front_of'
        self.relationship_classes[7] = 'on_the_side_of'
        self.relationship_classes[10] = 'covered_by'
        self.relationship_classes[11] = 'drinking_from'
        self.relationship_classes[13] = 'have_it_on_the_back'
        self.relationship_classes[15] = 'leaning_on'
        self.relationship_classes[16] = 'lying_on'
        self.relationship_classes[17] = 'not_contacting'
        self.relationship_classes[18] = 'other_relationship'
        self.relationship_classes[19] = 'sitting_on'
        self.relationship_classes[20] = 'standing_on'
        self.relationship_classes[25] = 'writing_on'
        
        
        self.attention_relationships = self.relationship_classes[0:3]
        self.spatial_relationships = self.relationship_classes[3:9]
        self.contacting_relationships = self.relationship_classes[9:]

        print(f'[Action Genome] Object Classes: {len(self.object_classes)}')
        print(f'Relationship Classes: {len(self.relationship_classes)},   att: {len(self.attention_relationships)}, spatial: {len(self.spatial_relationships)}, contact: {len(self.contacting_relationships)}\n')
        
        # for video_id in self.video_name_list:
        #     video_path = os.path.join(self.data_root_path, video_id, 'frame')
            # for frame_id in frame_list:
            #     frame_path = os.path.join(frame_dir_path, frame_id)
            #     self.data_list.append(frame_path)                
        
        print(f'Total Video: {len(self.video_list)}')
        
    def __getitem__(self, index):

        video_name = self.video_list[index]
        frame_names = os.listdir(os.path.join(self.data_root_path, video_name, 'frame'))
        processed_ims = []
        im_scales = []

        for idx, name in enumerate(frame_names):
            im = imread(os.path.join(self.data_root_path, video_name, 'frame', name),cv2.IMREAD_UNCHANGED) # channel h,w,3
            im, im_scale = prep_im_for_blob(im, [[[102.9801, 115.9465, 122.7717]]], 600, 1000) #cfg.PIXEL_MEANS, target_size, cfg.TRAIN.MAX_SIZE
            im_scales.append(im_scale)
            processed_ims.append(im)

        blob = im_list_to_blob(processed_ims)
        im_info = np.array([[blob.shape[1], blob.shape[2], im_scales[0]]],dtype=np.float32)
        im_info = torch.from_numpy(im_info).repeat(blob.shape[0], 1)
        img_tensor = torch.from_numpy(blob)
        img_tensor = img_tensor.permute(0, 3, 1, 2)

        gt_boxes = torch.zeros([img_tensor.shape[0], 1, 5])
        # gt_boxes = self.gt_annotations[]
        num_boxes = torch.zeros([img_tensor.shape[0]], dtype=torch.int64)
        video_size = im.shape

        return img_tensor, im_info, gt_boxes, num_boxes, video_name, video_size
    
    def get_viz_img(self, index):
    
        video_name = self.video_list[index]
        frame_names = os.listdir(os.path.join(self.data_root_path, video_name, 'frame'))
        processed_ims = []
        for idx, name in enumerate(frame_names):
            im = imread(os.path.join(self.data_root_path, video_name, 'frame', name),cv2.IMREAD_UNCHANGED) # channel h,w,3
            processed_ims.append(im)
        return np.array(processed_ims)


    def __len__(self):
        return len(self.video_list)

def cuda_collate_fn(batch):
    """
    don't need to zip the tensor

    """
    return batch[0]

def im_list_to_blob(ims):
    """
    Convert a list of images into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
    for i in range(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im

    return blob

def prep_im_for_blob(im, pixel_means, target_size, max_size):
    """ Mean subtract and scale an image for use in a blob. """

    im = im.astype(np.float32, copy=False)
    im -= pixel_means
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    
    # Prevent the biggest axis from being more than MAX_SIZE
    """
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = imresize(im, im_scale)
    """
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)

    return im, im_scale

if __name__ == '__main__':
    print(f'========== Action Genome Dataset ==========')
    dataset = KCC_AG()
    img_tensor, im_info, gt_boxes, num_boxes, index, video_size = dataset[100] #67
    print(f'\nimg: {img_tensor.shape},  img_scale: x{im_info[0,-1]:.4f}')
    print(f'gt_boxes: {gt_boxes.shape},  num_boxes: {num_boxes.shape[0]:.0f}')
    print(f'\n=============== Complete ===============')