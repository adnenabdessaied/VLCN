# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from cfgs.base_cfgs import Cfgs
from core.exec import Execution
import argparse, yaml, os

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    '''
    Parse input arguments
    '''
    parser = argparse.ArgumentParser(description='VLCN Args')

    parser.add_argument('--RUN', dest='RUN_MODE',
                      default='train',
                      choices=['train', 'val', 'test'],
                      help='{train, val, test}',
                      type=str)  # , required=True)

    parser.add_argument('--MODEL', dest='MODEL',
                      choices=['small', 'large'],
                      help='{small, large}',
                      default='small', type=str)

    parser.add_argument('--OPTIM', dest='OPTIM',
                      choices=['adam', 'rmsprop'],
                      help='The optimizer',
                      default='rmsprop', type=str)

    parser.add_argument('--SPLIT', dest='TRAIN_SPLIT',
                      choices=['train', 'train+val'],
                      help="set training split, "
                           "eg.'train', 'train+val'"
                           "set 'train' can trigger the "
                           "eval after every epoch",
                      default='train',
                      type=str)

    parser.add_argument('--EVAL_EE', dest='EVAL_EVERY_EPOCH',
                      default=True,
                      help='set True to evaluate the '
                           'val split when an epoch finished'
                           "(only work when train with "
                           "'train' split)",
                      type=bool)

    parser.add_argument('--SAVE_PRED', dest='TEST_SAVE_PRED',
                      help='set True to save the '
                           'prediction vectors'
                           '(only work in testing)',
                      default=False,
                      type=bool)

    parser.add_argument('--BS', dest='BATCH_SIZE',
                      help='batch size during training',
                      default=64,
                      type=int)

    parser.add_argument('--MAX_EPOCH', dest='MAX_EPOCH',
                      default=30,
                      help='max training epoch',
                      type=int)

    parser.add_argument('--PRELOAD', dest='PRELOAD',
                      help='pre-load the features into memory'
                           'to increase the I/O speed',
                      default=False,
                      type=bool)

    parser.add_argument('--GPU', dest='GPU',
                      help="gpu select, eg.'0, 1, 2'",
                      default='0',
                      type=str)

    parser.add_argument('--SEED', dest='SEED',
                      help='fix random seed',
                      default=42,
                      type=int)

    parser.add_argument('--VERSION', dest='VERSION',
                      help='version control',
                      default='1.0.0',
                      type=str)

    parser.add_argument('--RESUME', dest='RESUME',
                      default=False,
                      help='resume training',
                      type=str2bool)

    parser.add_argument('--CKPT_V', dest='CKPT_VERSION',
                      help='checkpoint version',
                      type=str)

    parser.add_argument('--CKPT_E', dest='CKPT_EPOCH',
                      help='checkpoint epoch',
                      type=int)

    parser.add_argument('--CKPT_PATH', dest='CKPT_PATH',
                      help='load checkpoint path, we '
                           'recommend that you use '
                           'CKPT_VERSION and CKPT_EPOCH '
                           'instead',
                      type=str)

    parser.add_argument('--ACCU', dest='GRAD_ACCU_STEPS',
                      help='reduce gpu memory usage',
                      type=int)

    parser.add_argument('--NW', dest='NUM_WORKERS',
                      help='multithreaded loading',
                      default=0,
                      type=int)

    parser.add_argument('--PINM', dest='PIN_MEM',
                      help='use pin memory',
                      type=bool)

    parser.add_argument('--VERB', dest='VERBOSE',
                      help='verbose print',
                      type=bool)

    parser.add_argument('--DATA_PATH', dest='DATASET_PATH',
                      default='/projects/abdessaied/data/MSRVTT-QA/',
                      help='Dataset root path',
                      type=str)

    parser.add_argument('--EXP_NAME', dest='EXP_NAME',
                      help='The name of the experiment',
                      default="test",
                      type=str)

    parser.add_argument('--DEBUG', dest='DEBUG',
                      help='Triggeres debug mode: small fractions of the data are loaded ',
                      default='0',
                      type=str2bool)

    parser.add_argument('--ENABLE_TIME_MONITORING', dest='ENABLE_TIME_MONITORING',
                      help='Triggeres time monitoring when training',
                      default='0',
                      type=str2bool)

    parser.add_argument('--MODEL_TYPE', dest='MODEL_TYPE',
                      help='The model type to be used\n 1: VLCN \n 2:VLCN-FLF \n 3: VLCN+LSTM \n 4: MCAN',
                      default=1,
                      type=int)

    parser.add_argument('--PRETRAINED_PATH', dest='PRETRAINED_PATH',
                      help='Pretrained weights on msvd',
                      default='-',
                      type=str)

    parser.add_argument('--TEST_EPOCH', dest='TEST_EPOCH',
               help='',
               default=7,
               type=int)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
     args = parse_args()
     os.chdir(os.path.dirname(os.path.abspath(__file__)))
     __C = Cfgs(args.EXP_NAME, args.DATASET_PATH)
     args_dict = __C.parse_to_dict(args)

     cfg_file = "cfgs/{}_model.yml".format(args.MODEL)
     with open(cfg_file, 'r') as f:
          yaml_dict = yaml.load(f)

     args_dict = {**yaml_dict, **args_dict}
     
     __C.add_args(args_dict)
     __C.proc()

     print('Hyper Parameters:')
     print(__C)

     __C.check_path()
     os.environ['CUDA_VISIBLE_DEVICES'] = __C.GPU

     execution = Execution(__C)
     execution.run(__C.RUN_MODE)

     #execution.run('test', epoch=__C.TEST_EPOCH)
