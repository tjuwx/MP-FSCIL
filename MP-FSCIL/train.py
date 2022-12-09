import argparse
import importlib
from utils import *

MODEL_DIR=None
DATA_DIR = 'data/'
# PROJECT='base'

def get_command_line_parser():
    parser = argparse.ArgumentParser()

    # about dataset and network
    parser.add_argument('-dataset', type=str, default='cub200',
                        choices=['mini_imagenet', 'cub200', 'cifar100'])
    parser.add_argument('-dataroot', type=str, default=DATA_DIR)

    # about pre-training
    parser.add_argument('-pre_epochs', type=int, default=100)
    parser.add_argument('-pre_lr', type=float, default=0.1)
    parser.add_argument('-pre_schedule', type=str, default='Milestone',
                        choices=['Step', 'Milestone'])
    parser.add_argument('-pre_milestones', nargs='+', type=int, default=[40, 70])
    parser.add_argument('-pre_step', type=int, default=40)
    parser.add_argument('-temperature', type=int, default=16)
    parser.add_argument('-pre_batch_size', type=int, default=128)
    parser.add_argument('-batch_size_new', type=int, default=0, help='set 0 will use all the availiable training image for new')
    parser.add_argument('-test_batch_size', type=int, default=100)
    parser.add_argument('-num', type=int, default=2)
    
    # about meta-training
    parser.add_argument('-meta_episode', type=int, default=100)
    parser.add_argument('-meta_shot', type=int, default=3)
    parser.add_argument('-meta_way', type=int, default=20)
    parser.add_argument('-meta_query', type=int, default=10)
    parser.add_argument('-meta_lr', type=float, default=0.0005)
    parser.add_argument('-meta_schedule', type=str, default='Step',
                        choices=['Step', 'Milestone'])
    parser.add_argument('-meta_milestones', nargs='+', type=int, default=[60, 70])
    parser.add_argument('-meta_step', type=int, default=20)
    parser.add_argument('-start_session', type=int, default=0)
    parser.add_argument('-model_dir', type=str, default=MODEL_DIR, help='loading model parameter from a specific dir')
    parser.add_argument('-set_no_val', action='store_true', help='set validation using test set or no validation')

    # about training
    parser.add_argument('-gpu', default='0,1,2,3')
    parser.add_argument('-num_workers', type=int, default=8)
    parser.add_argument('-seed', type=int, default=1)
    parser.add_argument('-debug', action='store_true')

    return parser


if __name__ == '__main__':
    parser = get_command_line_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    pprint(vars(args))
    args.num_gpu = set_gpu(args)
    trainer = importlib.import_module('models.mp.fscil_trainer').FSCILTrainer(args)
    trainer.train()




