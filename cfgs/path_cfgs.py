# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

import os
 
class PATH:
    def __init__(self, EXP_NAME, DATASET_PATH):
        # name of the experiment
        self.EXP_NAME = EXP_NAME

        # Dataset root path
        self.DATASET_PATH = DATASET_PATH 

        # Bottom up features root path
        self.FRAMES = os.path.join(DATASET_PATH, 'frame_feat/')
        self.CLIPS = os.path.join(DATASET_PATH, 'clip_feat/')


    def init_path(self):
        self.QA_PATH = {
            'train': self.DATASET_PATH + 'train_qa.json',
            'val': self.DATASET_PATH + 'val_qa.json',
            'test': self.DATASET_PATH + 'test_qa.json',
        }
        self.C3D_PATH = self.DATASET_PATH + 'c3d.pickle'

        if self.EXP_NAME not in os.listdir('./'):
            os.mkdir('./' + self.EXP_NAME)
            os.mkdir('./' + self.EXP_NAME + '/results')
        self.RESULT_PATH = './' + self.EXP_NAME + '/results/result_test/'
        self.PRED_PATH = './' + self.EXP_NAME + '/results/pred/'
        self.CACHE_PATH = './' + self.EXP_NAME + '/results/cache/'
        self.LOG_PATH = './' + self.EXP_NAME + '/results/log/'
        self.TB_PATH = './' + self.EXP_NAME + '/results/tensorboard/'
        self.CKPTS_PATH = './' + self.EXP_NAME + '/ckpts/'

        if 'result_test' not in os.listdir('./' + self.EXP_NAME + '/results'):
            os.mkdir('./' + self.EXP_NAME + '/results/result_test/')

        if 'pred' not in os.listdir('./' + self.EXP_NAME + '/results'):
            os.mkdir('./' + self.EXP_NAME + '/results/pred/')

        if 'cache' not in os.listdir('./' + self.EXP_NAME + '/results'):
            os.mkdir('./' + self.EXP_NAME + '/results/cache')

        if 'log' not in os.listdir('./' + self.EXP_NAME + '/results'):
            os.mkdir('./' + self.EXP_NAME + '/results/log')

        if 'tensorboard' not in os.listdir('./' + self.EXP_NAME + '/results'):
            os.mkdir('./' + self.EXP_NAME + '/results/tensorboard')

        if 'ckpts' not in os.listdir('./' + self.EXP_NAME):
            os.mkdir('./' + self.EXP_NAME + '/ckpts')


    def check_path(self):
        raise NotImplementedError

