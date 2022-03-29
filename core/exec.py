# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from core.data.dataset import VideoQA_Dataset
from core.model.net import Net1, Net2, Net3, Net4
from core.model.optim import get_optim, adjust_lr
from core.metrics import get_acc
from tqdm import tqdm
from core.data.utils import shuffle_list

import os, json, torch, datetime, pickle, copy, shutil, time, math
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
from tensorboardX import SummaryWriter
from torch.autograd import Variable as var

class Execution:
    def __init__(self, __C):
        self.__C = __C
        print('Loading training set ........')
        __C_train = copy.deepcopy(self.__C)
        setattr(__C_train, 'RUN_MODE', 'train')
        self.dataset = VideoQA_Dataset(__C_train)

        self.dataset_eval = None
        if self.__C.EVAL_EVERY_EPOCH:
            __C_eval = copy.deepcopy(self.__C)
            setattr(__C_eval, 'RUN_MODE', 'val')

            print('Loading validation set for per-epoch evaluation ........')
            self.dataset_eval = VideoQA_Dataset(__C_eval)
            self.dataset_eval.ans_list = self.dataset.ans_list
            self.dataset_eval.ans_to_ix, self.dataset_eval.ix_to_ans = self.dataset.ans_to_ix, self.dataset.ix_to_ans
            self.dataset_eval.token_to_ix, self.dataset_eval.pretrained_emb = self.dataset.token_to_ix, self.dataset.pretrained_emb
        
        __C_test = copy.deepcopy(self.__C)
        setattr(__C_test, 'RUN_MODE', 'test')

        self.dataset_test = VideoQA_Dataset(__C_test)
        self.dataset_test.ans_list = self.dataset.ans_list
        self.dataset_test.ans_to_ix, self.dataset_test.ix_to_ans = self.dataset.ans_to_ix, self.dataset.ix_to_ans
        self.dataset_test.token_to_ix, self.dataset_test.pretrained_emb = self.dataset.token_to_ix, self.dataset.pretrained_emb
        
        self.writer = SummaryWriter(self.__C.TB_PATH)

    def train(self, dataset, dataset_eval=None):
        # Obtain needed information
        data_size = dataset.data_size
        token_size = dataset.token_size
        ans_size = dataset.ans_size
        pretrained_emb = dataset.pretrained_emb
        net = self.construct_net(self.__C.MODEL_TYPE)
        if os.path.isfile(self.__C.PRETRAINED_PATH) and self.__C.MODEL_TYPE == 11:
            print('Loading pretrained DNC-weigths')
            net.load_pretrained_weights()
        net.cuda()
        net.train()

        # Define the multi-gpu training if needed
        if self.__C.N_GPU > 1:
            net = nn.DataParallel(net, device_ids=self.__C.DEVICES)

        # Define the binary cross entropy loss
        # loss_fn = torch.nn.BCELoss(size_average=False).cuda()
        loss_fn = torch.nn.BCELoss(reduction='sum').cuda()
        # Load checkpoint if resume training
        if self.__C.RESUME:
            print(' ========== Resume training')

            if self.__C.CKPT_PATH is not None:
                print('Warning: you are now using CKPT_PATH args, '
                      'CKPT_VERSION and CKPT_EPOCH will not work')

                path = self.__C.CKPT_PATH
            else:
                path = self.__C.CKPTS_PATH + \
                       'ckpt_' + self.__C.CKPT_VERSION + \
                       '/epoch' + str(self.__C.CKPT_EPOCH) + '.pkl'

            # Load the network parameters
            print('Loading ckpt {}'.format(path))
            ckpt = torch.load(path)
            print('Finish!')
            net.load_state_dict(ckpt['state_dict'])

            # Load the optimizer paramters
            optim = get_optim(self.__C, net, data_size, ckpt['optim'], lr_base=ckpt['lr_base'])
            optim._step = int(data_size / self.__C.BATCH_SIZE * self.__C.CKPT_EPOCH)
            optim.optimizer.load_state_dict(ckpt['optimizer'])

            start_epoch = self.__C.CKPT_EPOCH

        else:
            if ('ckpt_' + self.__C.VERSION) in os.listdir(self.__C.CKPTS_PATH):
                shutil.rmtree(self.__C.CKPTS_PATH + 'ckpt_' + self.__C.VERSION)

            os.mkdir(self.__C.CKPTS_PATH + 'ckpt_' + self.__C.VERSION)

            optim = get_optim(self.__C, net, data_size, self.__C.OPTIM)
            start_epoch = 0

        loss_sum = 0
        named_params = list(net.named_parameters())
        grad_norm = np.zeros(len(named_params))

        # Define multi-thread dataloader
        if self.__C.SHUFFLE_MODE in ['external']:
            dataloader = Data.DataLoader(
                dataset,
                batch_size=self.__C.BATCH_SIZE,
                shuffle=False,
                num_workers=self.__C.NUM_WORKERS,
                pin_memory=self.__C.PIN_MEM,
                drop_last=True
            )
        else:
            dataloader = Data.DataLoader(
                dataset,
                batch_size=self.__C.BATCH_SIZE,
                shuffle=True,
                num_workers=self.__C.NUM_WORKERS,
                pin_memory=self.__C.PIN_MEM,
                drop_last=True
            )

        # Training script
        for epoch in range(start_epoch, self.__C.MAX_EPOCH):

            # Save log information
            logfile = open(
                self.__C.LOG_PATH +
                'log_run_' + self.__C.VERSION + '.txt',
                'a+'
            )
            logfile.write(
                'nowTime: ' +
                datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
                '\n'
            )
            logfile.close()

            # Learning Rate Decay
            if epoch in self.__C.LR_DECAY_LIST:
                adjust_lr(optim, self.__C.LR_DECAY_R)
         
            # Externally shuffle
            if self.__C.SHUFFLE_MODE == 'external':
                shuffle_list(dataset.ans_list)

            time_start = time.time()
            # Iteration
            for step, (
                    ques_ix_iter,
                    frames_feat_iter,
                    clips_feat_iter,
                    ans_iter,
                    _,
                    _,
                    _,
                    _
            ) in enumerate(dataloader):

                ques_ix_iter = ques_ix_iter.cuda()
                frames_feat_iter = frames_feat_iter.cuda()
                clips_feat_iter = clips_feat_iter.cuda()
                ans_iter = ans_iter.cuda()

                optim.zero_grad()

                for accu_step in range(self.__C.GRAD_ACCU_STEPS):

                    sub_frames_feat_iter = \
                        frames_feat_iter[accu_step * self.__C.SUB_BATCH_SIZE:
                                      (accu_step + 1) * self.__C.SUB_BATCH_SIZE]
                    sub_clips_feat_iter = \
                        clips_feat_iter[accu_step * self.__C.SUB_BATCH_SIZE:
                                      (accu_step + 1) * self.__C.SUB_BATCH_SIZE]
                    sub_ques_ix_iter = \
                        ques_ix_iter[accu_step * self.__C.SUB_BATCH_SIZE:
                                     (accu_step + 1) * self.__C.SUB_BATCH_SIZE]
                    sub_ans_iter = \
                        ans_iter[accu_step * self.__C.SUB_BATCH_SIZE:
                                 (accu_step + 1) * self.__C.SUB_BATCH_SIZE]
                    
                    pred = net(
                        sub_frames_feat_iter,
                        sub_clips_feat_iter,
                        sub_ques_ix_iter
                    )
                    
                    loss = loss_fn(pred, sub_ans_iter)

                    # only mean-reduction needs be divided by grad_accu_steps
                    # removing this line wouldn't change our results because the speciality of Adam optimizer,
                    # but would be necessary if you use SGD optimizer.
                    # loss /= self.__C.GRAD_ACCU_STEPS
                    # start_backward = time.time()
                    loss.backward()

                    if self.__C.VERBOSE:
                        if dataset_eval is not None:
                            mode_str = self.__C.SPLIT['train'] + '->' + self.__C.SPLIT['val']
                        else:
                            mode_str = self.__C.SPLIT['train'] + '->' + self.__C.SPLIT['test']
                        
                        # logging
                        
                        self.writer.add_scalar(
                            'train/loss',
                            loss.cpu().data.numpy() / self.__C.SUB_BATCH_SIZE,
                            global_step=step + epoch * math.ceil(data_size / self.__C.BATCH_SIZE))
                        
                        self.writer.add_scalar(
                            'train/lr',
                            optim._rate,
                            global_step=step + epoch * math.ceil(data_size / self.__C.BATCH_SIZE))
                       
                        print("\r[exp_name %s][version %s][epoch %2d][step %4d/%4d][%s] loss: %.4f, lr: %.2e" % (
                            self.__C.EXP_NAME,
                            self.__C.VERSION,
                            epoch + 1,
                            step,
                            int(data_size / self.__C.BATCH_SIZE),
                            mode_str,
                            loss.cpu().data.numpy() / self.__C.SUB_BATCH_SIZE,
                            optim._rate,
                        ), end='          ')

                # Gradient norm clipping
                if self.__C.GRAD_NORM_CLIP > 0:
                    nn.utils.clip_grad_norm_(
                        net.parameters(),
                        self.__C.GRAD_NORM_CLIP
                    )

                # Save the gradient information
                for name in range(len(named_params)):
                    norm_v = torch.norm(named_params[name][1].grad).cpu().data.numpy() \
                        if named_params[name][1].grad is not None else 0
                    grad_norm[name] += norm_v * self.__C.GRAD_ACCU_STEPS

                optim.step()

            time_end = time.time()
            print('Finished in {}s'.format(int(time_end-time_start)))

            epoch_finish = epoch + 1

            # Save checkpoint
            state = {
                'state_dict': net.state_dict(),
                'optimizer': optim.optimizer.state_dict(),
                'lr_base': optim.lr_base,
                'optim': optim.lr_base,            }                

            torch.save(
                state,
                self.__C.CKPTS_PATH +
                'ckpt_' + self.__C.VERSION +
                '/epoch' + str(epoch_finish) +
                '.pkl'
            )

            # Logging
            logfile = open(
                self.__C.LOG_PATH +
                'log_run_' + self.__C.VERSION + '.txt',
                'a+'
            )
            logfile.write(
                'epoch = ' + str(epoch_finish) +
                '  loss = ' + str(loss_sum / data_size) +
                '\n' +
                'lr = ' + str(optim._rate) +
                '\n\n'
            )
            logfile.close()

            # Eval after every epoch
            if dataset_eval is not None:
                self.eval(
                    net,
                    dataset_eval,
                    self.writer,
                    epoch,
                    valid=True,
                )

            loss_sum = 0
            grad_norm = np.zeros(len(named_params))


    # Evaluation
    def eval(self, net, dataset, writer, epoch, valid=False):
        
        ans_ix_list = []
        pred_list = []
        q_type_list = []
        q_bin_list = []
        ans_rarity_list = []

        ans_qtype_dict  = {'what': [], 'who': [], 'how': [], 'when': [], 'where': []}
        pred_qtype_dict = {'what': [], 'who': [], 'how': [], 'when': [], 'where': []}


        ans_qlen_bin_dict  = {'1-3': [], '4-8': [], '9-15': []}
        pred_qlen_bin_dict = {'1-3': [], '4-8': [], '9-15': []}
    
        ans_ans_rarity_dict  = {'0-99': [], '100-299': [], '300-999': []}
        pred_ans_rarity_dict = {'0-99': [], '100-299': [], '300-999': []}

        data_size = dataset.data_size
        
        net.eval()

        if self.__C.N_GPU > 1:
            net = nn.DataParallel(net, device_ids=self.__C.DEVICES)

        dataloader = Data.DataLoader(
            dataset,
            batch_size=self.__C.EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=self.__C.NUM_WORKERS,
            pin_memory=True
        )

        for step, (
                ques_ix_iter,
                frames_feat_iter,
                clips_feat_iter,
                _,
                ans_iter,
                q_type,
                qlen_bin,
                ans_rarity
            ) in enumerate(dataloader):
            print("\rEvaluation: [step %4d/%4d]" % (
                step,
                int(data_size / self.__C.EVAL_BATCH_SIZE),
            ), end='          ')
            ques_ix_iter = ques_ix_iter.cuda()
            frames_feat_iter = frames_feat_iter.cuda()
            clips_feat_iter = clips_feat_iter.cuda()
            with torch.no_grad():

                pred = net(
                    frames_feat_iter,
                    clips_feat_iter,
                    ques_ix_iter
                )
                
            pred_np = pred.cpu().data.numpy()
            pred_argmax = np.argmax(pred_np, axis=1)
            pred_list.extend(pred_argmax)
            ans_ix_list.extend(ans_iter.tolist())
            q_type_list.extend(q_type.tolist())
            q_bin_list.extend(qlen_bin.tolist())
            ans_rarity_list.extend(ans_rarity.tolist())

        print('')

        assert len(pred_list) == len(ans_ix_list) == len(q_type_list) == len(q_bin_list) == len(ans_rarity_list)
        pred_list = [dataset.ix_to_ans[pred] for pred in pred_list]
        ans_ix_list = [dataset.ix_to_ans[ans] for ans in ans_ix_list]

        # Run validation script
        scores_per_qtype = {
            'what': {},
            'who': {},
            'how': {},
            'when': {},
            'where': {},
            }
        scores_per_qlen_bin = {
            '1-3': {},
            '4-8': {},
            '9-15': {},
            }
        scores_ans_rarity_dict = {
            '0-99': {},
            '100-299': {},
            '300-999': {}
            }

        if valid:
            # create vqa object and vqaRes object
            for pred, ans, q_type in zip(pred_list, ans_ix_list, q_type_list):
                pred_qtype_dict[dataset.idx_to_qtypes[q_type]].append(pred)
                ans_qtype_dict[dataset.idx_to_qtypes[q_type]].append(ans)

            print('----------------- Computing scores -----------------')
            acc = get_acc(ans_ix_list, pred_list)
            print('----------------- Overall  -----------------')
            print('acc: {}'.format(acc))
            writer.add_scalar('acc/overall', acc, global_step=epoch)

            for q_type in scores_per_qtype:
                print('----------------- Computing "{}" q-type scores -----------------'.format(q_type))
                # acc, wups_0, wups_1 = get_scores(
                #     ans_ix_dict[q_type], pred_ix_dict[q_type])
                acc = get_acc(ans_qtype_dict[q_type], pred_qtype_dict[q_type])
                print('acc: {}'.format(acc))
                writer.add_scalar(
                    'acc/{}'.format(q_type), acc, global_step=epoch)
        else:
            for pred, ans, q_type, qlen_bin, a_rarity in zip(
                pred_list, ans_ix_list, q_type_list, q_bin_list, ans_rarity_list):

                pred_qtype_dict[dataset.idx_to_qtypes[q_type]].append(pred)
                ans_qtype_dict[dataset.idx_to_qtypes[q_type]].append(ans)

                pred_qlen_bin_dict[dataset.idx_to_qlen_bins[qlen_bin]].append(pred)
                ans_qlen_bin_dict[dataset.idx_to_qlen_bins[qlen_bin]].append(ans)

                pred_ans_rarity_dict[dataset.idx_to_ans_rare[a_rarity]].append(pred)
                ans_ans_rarity_dict[dataset.idx_to_ans_rare[a_rarity]].append(ans)

            print('----------------- Computing overall scores -----------------')
            acc = get_acc(ans_ix_list, pred_list)

            print('----------------- Overall  -----------------')
            print('acc:{}'.format(acc))


            print('----------------- Computing q-type scores -----------------')
            for q_type in scores_per_qtype:
                acc = get_acc(ans_qtype_dict[q_type], pred_qtype_dict[q_type])
                print('                 {}                 '.format(q_type))
                print('acc:{}'.format(acc))

            print('----------------- Computing qlen-bins scores -----------------')
            for qlen_bin in scores_per_qlen_bin:

                acc = get_acc(ans_qlen_bin_dict[qlen_bin], pred_qlen_bin_dict[qlen_bin])
                print('                 {}                 '.format(qlen_bin))
                print('acc:{}'.format(acc))

            print('----------------- Computing ans-rarity scores -----------------')
            for a_rarity in scores_ans_rarity_dict:
                acc = get_acc(ans_ans_rarity_dict[a_rarity], pred_ans_rarity_dict[a_rarity])
                print('                 {}                 '.format(a_rarity))
                print('acc:{}'.format(acc))
        net.train()

    def construct_net(self, model_type):
        if model_type == 1:
            net = Net1(
                self.__C,
                self.dataset.pretrained_emb,
                self.dataset.token_size,
                self.dataset.ans_size
            )
        elif model_type == 2:
            net = Net2(
                self.__C,
                self.dataset.pretrained_emb,
                self.dataset.token_size,
                self.dataset.ans_size
            )
        elif model_type == 3:
            net = Net3(
                self.__C,
                self.dataset.pretrained_emb,
                self.dataset.token_size,
                self.dataset.ans_size
            )
        elif model_type == 4:
            net = Net4(
                self.__C,
                self.dataset.pretrained_emb,
                self.dataset.token_size,
                self.dataset.ans_size
            )
        else:
            raise ValueError('Net{} is not supported'.format(model_type))
        return net
        
    def run(self, run_mode, epoch=None):
        self.set_seed(self.__C.SEED)
        if run_mode == 'train':
            self.empty_log(self.__C.VERSION)
            self.train(self.dataset, self.dataset_eval)

        elif run_mode == 'val':
            self.eval(self.dataset, valid=True)

        elif run_mode == 'test':
            net = self.construct_net(self.__C.MODEL_TYPE) 
            assert epoch is not None
            path = self.__C.CKPTS_PATH + \
                   'ckpt_' + self.__C.VERSION + \
                   '/epoch' + str(epoch) + '.pkl'
            print('Loading ckpt {}'.format(path))
            state_dict = torch.load(path)['state_dict']
            net.load_state_dict(state_dict)
            net.cuda()
            self.eval(net, self.dataset_test, self.writer, 0)

        else:
            exit(-1)

    def set_seed(self, seed):
        """Sets the seed for reproducibility. 
        Args:
            seed (int): The seed used
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        print('\nSeed set to {}...\n'.format(seed))

    def empty_log(self, version):
        print('Initializing log file ........')
        if (os.path.exists(self.__C.LOG_PATH + 'log_run_' + version + '.txt')):
            os.remove(self.__C.LOG_PATH + 'log_run_' + version + '.txt')
        print('Finished!')
        print('')
