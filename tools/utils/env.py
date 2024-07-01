import os
import torch
import warnings
import numpy as np

def set_seed_and_igwarn(seed=1123):
    warnings.filterwarnings(action='ignore')
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

    seed = 1123              # 원하는 시드 값
    torch.manual_seed(seed)  # 전체 랜덤 시드 설정
    return seed

def set_train_dir(path, mode):
    if not os.path.exists(path):
        os.mkdir(path)
    path = path+mode+'/'
    
    model_path = os.path.join(path, "models")
    tf_path = os.path.join(path, "tensorboard")
    
    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    if not os.path.exists(tf_path):
        os.mkdir(tf_path)
        
    log = open(path+'log_train.txt', mode = 'w')
    log.write('*'*60+'\n')
    
    log_val = open(path+'log_val.txt', mode = 'w')
    log_val.write('*'*60+'\n')
    
    return log, log_val, model_path, tf_path