import glob, os, json, pickle
import numpy as np
from collections import defaultdict

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from core.data.utils import tokenize, ans_stat, proc_ques, qlen_to_key, ans_to_key


class VideoQA_Dataset(Dataset):
    def __init__(self, __C):
        super(VideoQA_Dataset, self).__init__()
        self.__C = __C
        self.ans_size = __C.NUM_ANS
        # load raw data
        with open(__C.QA_PATH[__C.RUN_MODE], 'r') as f:
            self.raw_data = json.load(f)
        self.data_size = len(self.raw_data)

        splits = __C.SPLIT[__C.RUN_MODE].split('+')
        
        frames_list = glob.glob(__C.FRAMES + '*.pt')
        clips_list = glob.glob(__C.CLIPS + '*.pt')
        if 'msvd' in self.C.DATASET_PATH.lower():
            vid_ids = [int(s.split('/')[-1].split('.')[0][3:]) for s in frames_list]
        else:
            vid_ids = [int(s.split('/')[-1].split('.')[0][5:]) for s in frames_list]
        self.frames_dict = {k: v for (k,v) in zip(vid_ids, frames_list)}
        self.clips_dict = {k: v for (k,v) in zip(vid_ids, clips_list)}
        del frames_list, clips_list

        q_list = []
        a_list = []
        a_dict = defaultdict(lambda: 0)
        for split in ['train', 'val']:
            with  open(__C.QA_PATH[split], 'r') as f:
                qa_data = json.load(f)
            for d in qa_data:
                q_list.append(d['question'])
                a_list = d['answer']
                if d['answer'] not in a_dict:
                    a_dict[d['answer']] = 1
                else:
                    a_dict[d['answer']] += 1

        top_answers = sorted(a_dict, key=a_dict.get, reverse=True)
        self.qlen_bins_to_idx = {
            '1-3': 0,
            '4-8': 1,
            '9-15': 2,
        }
        self.ans_rare_to_idx = {
            '0-99': 0,
            '100-299': 1,
            '300-999': 2,

        }
        self.qtypes_to_idx = {
            'what': 0,
            'who': 1,
            'how': 2,
            'when': 3,
            'where': 4,
        }

        if __C.RUN_MODE == 'train':
            self.ans_list = top_answers[:self.ans_size]

            self.ans_to_ix, self.ix_to_ans = ans_stat(self.ans_list)

            self.token_to_ix, self.pretrained_emb = tokenize(q_list, __C.USE_GLOVE)
            self.token_size = self.token_to_ix.__len__()
            print('== Question token vocab size:', self.token_size)
    
        self.idx_to_qtypes = {v: k for (k, v) in self.qtypes_to_idx.items()}
        self.idx_to_qlen_bins = {v: k for (k, v) in self.qlen_bins_to_idx.items()}
        self.idx_to_ans_rare = {v: k for (k, v) in self.ans_rare_to_idx.items()}

    def __getitem__(self, idx):
        sample = self.raw_data[idx]
        ques = sample['question']
        q_type = self.qtypes_to_idx[ques.split(' ')[0]]
        ques_idx, qlen, _ = proc_ques(ques, self.token_to_ix, self.__C.MAX_TOKEN)
        qlen_bin = self.qlen_bins_to_idx[qlen_to_key(qlen)]

        answer = sample['answer']
        answer = self.ans_to_ix.get(answer, np.random.randint(0, high=len(self.ans_list)))
        ans_rarity = self.ans_rare_to_idx[ans_to_key(answer)]

        answer_one_hot = torch.zeros(self.ans_size)
        answer_one_hot[answer] = 1.0

        vid_id = sample['video_id']
        frames = torch.load(open(self.frames_dict[vid_id], 'rb')).cpu()
        clips = torch.load(open(self.clips_dict[vid_id], 'rb')).cpu()

        return torch.from_numpy(ques_idx).long(), frames, clips, answer_one_hot, torch.tensor(answer).long(), \
            torch.tensor(q_type).long(), torch.tensor(qlen_bin).long(), torch.tensor(ans_rarity).long()

    def __len__(self):
        return self.data_size
