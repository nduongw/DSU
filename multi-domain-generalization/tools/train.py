import argparse
import torch
import wandb

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer


def print_args(args, cfg):
    print('***************')
    print('** Arguments **')
    print('***************')
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print('{}: {}'.format(key, args.__dict__[key]))
    print('************')
    print('** Config **')
    print('************')
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head
    
    if args.cluster:
        cfg.CLUSTER = args.cluster

    if args.num_clusters:
        cfg.NUM_CLUSTERS = args.num_clusters
    #if args.uncertainty:
    cfg.MODEL.UNCERTAINTY = args.uncertainty

    #if args.pos:
    cfg.MODEL.POS = args.pos

def setup_cfg(args):
    cfg = get_cfg_default()
    reset_cfg(cfg, args)
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def main(args):
    cfg = setup_cfg(args)
    if args.wandb:
        if cfg.MODEL.BACKBONE.NAME == 'uresnet18':
            job_type = 'DSU'
        elif cfg.MODEL.BACKBONE.NAME == 'cresnet18':
            job_type = 'ConstStyle3'
            if cfg.CLUSTER == 'ot':
                job_type += '-OT'
            elif cfg.CLUSTER == 'barycenter':
                job_type += '-BC'
            
            if cfg.NUM_CLUSTERS > 1:
                job_type += f'-num_clusters_{cfg.NUM_CLUSTERS}'
                
        elif cfg.MODEL.BACKBONE.NAME == 'resnet18_ms_l12':
            job_type = 'MixStyle'
        elif cfg.MODEL.BACKBONE.NAME == 'curesnet18':
            job_type = 'CSU'
        elif cfg.TRAINER.NAME == 'RIDG':
            job_type = 'RIDG'
        else:
            job_type = 'Baseline'
        
        if cfg.MODEL.BACKBONE.PRETRAINED:
            job_type += '-pretrained'
            
        tracker = wandb.init(
            project = 'StyleDG',
            entity = 'aiotlab',
            config = args,
            group = f'{cfg.DATASET.NAME}',
            name = f'train={cfg.DATASET.SOURCE_DOMAINS}_test={cfg.DATASET.TARGET_DOMAINS}_type={args.option}',
            job_type = job_type
        )
        args.tracker = tracker

    if cfg.SEED >= 0:
        print('Setting fixed seed: {}'.format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    print_args(args, cfg)

    trainer = build_trainer(cfg, args)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    if not args.no_train:
        trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='', help='path to dataset')
    parser.add_argument(
        '--output-dir', type=str, default='', help='output directory'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default='',
        help='checkpoint directory (from which the training resumes)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help='only positive value enables a fixed seed'
    )
    parser.add_argument(
        '--source-domains',
        type=str,
        nargs='+',
        help='source domains for DA/DG'
    )
    parser.add_argument(
        '--target-domains',
        type=str,
        nargs='+',
        help='target domains for DA/DG'
    )
    parser.add_argument(
        '--transforms', type=str, nargs='+', help='data augmentation methods'
    )
    parser.add_argument(
        '--config-file', type=str, default='', help='path to config file'
    )
    parser.add_argument(
        '--dataset-config-file',
        type=str,
        default='',
        help='path to config file for dataset setup'
    )
    parser.add_argument(
        '--trainer', type=str, default='', help='name of trainer'
    )
    parser.add_argument(
        '--backbone', type=str, default='', help='name of CNN backbone'
    )
    parser.add_argument('--head', type=str, default='', help='name of head')
    parser.add_argument(
        '--eval-only', action='store_true', help='evaluation only'
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default='',
        help='load model from this directory for eval-only mode'
    )
    parser.add_argument(
        '--load-epoch',
        type=int,
        help='load model weights at this epoch for evaluation'
    )
    parser.add_argument(
        '--no-train', action='store_true', help='do not call trainer.train()'
    )
    parser.add_argument(
        'opts',
        default=None,
        nargs=argparse.REMAINDER,
        help='modify config options using the command-line'
    )
    parser.add_argument('--uncertainty', default=0.0, type=float)
    parser.add_argument('--pos', nargs='+', type=int, default=[],
                        help='pos for uncertainty')
    parser.add_argument('--wandb', default=1, type=int, help='visualize on Wandb')
    parser.add_argument('--option', default='', type=str, help='additional options')
    parser.add_argument('--update_interval', default=25, type=int, help='update cluster interval')
    parser.add_argument('--cluster', default='ot', type=str, help='cluster choosing method')
    parser.add_argument('--num_clusters', default = 3, type = int, help='number of clusters')
    args = parser.parse_args()
    main(args)
