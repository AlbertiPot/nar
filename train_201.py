import os
import time
import math
import random
import argparse
import torch.utils.data
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from dataset import NASBench201DataBase, NASBench201Dataset, SplitSubet201
from architecture import Bucket
from ranker import Transformer
from sampler import ArchSampler201
from utils.loss_ops import CrossEntropyLossSoft, RankLoss
from utils.optim import LRScheduler
from utils.metric import AverageMeter
from utils.setup import setup_seed, setup_logger
from utils.config import get_config
from utils.saver import save_checkpoint
from process import train_epoch_201, validate_201, evaluate_sampled_batch_201
from process.train_utils import init_tier_list


def get_args():
    parser = argparse.ArgumentParser(description='NAR Training for nasbench201')
    parser.add_argument('--config_file',
                        default='./config/config201.yml',
                        type=str,
                        help='training configuration')
    parser.add_argument('--data_path',
                        default='./data/nasbench201/nasbench201_with_edge_flops_and_params.json',
                        type=str,
                        help='Path to load data')
    parser.add_argument('--save_dir',
                        default='./output',
                        type=str,
                        help='Path to save output')

    args = parser.parse_args()

    return args


def build_arg_and_env(run_args):
    args = get_config(run_args.config_file)

    args.config_file = run_args.config_file
    args.data_path = run_args.data_path

    if not os.path.exists(run_args.save_dir):
        os.makedirs(run_args.save_dir)

    args.save_dir = os.path.join(
        run_args.save_dir,
        args.exp_name + '_' + time.strftime('%Y%m%d%H%M%S', time.localtime()))

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    return args


def main():

    run_args = get_args()
    args = build_arg_and_env(run_args)

    # setup logger
    logger = setup_logger(save_path=os.path.join(args.save_dir, "train.log"))
    logger.info(args)
    # setup tensorboard
    tb_writer = SummaryWriter(os.path.join(args.save_dir,'tensorboard'))

    # setup global seed
    setup_seed(seed=args.seed)
    logger.info('set global random seed = {}'.format(args.seed))

    # setup cuda device
    if args.is_cuda:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch('cpu')

    # build dataset
    if args.space == 'nasbench201':
        database = NASBench201DataBase(args.data_path)
        dataset = NASBench201Dataset(database, seed=args.seed)
        trainset = SplitSubet201(dataset, list(range(args.train_size)), args.ranker.n_tier)
        valset = SplitSubet201(dataset, list(range(args.train_size, args.train_size + args.val_size)), args.ranker.n_tier)

    # build dataloader
    train_dataloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.train_batch_size,
        shuffle=True,
        drop_last=getattr(args, 'drop_last', True),
        num_workers=args.data_loader_workers,
        pin_memory=True)

    val_dataloader = torch.utils.data.DataLoader(
        valset,
        batch_size=args.val_batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.data_loader_workers,
        pin_memory=True)

    # build loss
    criterion = CrossEntropyLossSoft().cuda(device)
    if args.aux_loss:
        aux_criterion = RankLoss().cuda(device)

    # build model
    logger.info('Building model with {}'.format(args.ranker))
    ranker = Transformer(
        n_tier=args.ranker.n_tier,
        n_arch_patch=args.ranker.n_arch_patch,
        d_patch=args.ranker.d_patch,
        d_patch_vec=args.ranker.d_patch_vec,
        d_model=args.ranker.d_model,
        d_ffn_inner=args.ranker.d_ffn_inner,
        d_tier_prj_inner=args.ranker.d_tier_prj_inner,
        n_layers=args.ranker.n_layers,
        n_head=args.ranker.n_head,
        d_k=args.ranker.d_k,
        d_v=args.ranker.d_v,
        dropout=args.ranker.dropout,
        n_position=args.ranker.n_position,
        d_val_acc_prj_inner = args.ranker.d_val_acc_prj_inner,
        scale_prj=args.ranker.scale_prj)
    ranker.cuda(device)

    # build optimizer and lr_scheduler
    logger.info('Building optimizer and lr_scheduler')
    optimizer = optim.AdamW(
        ranker.parameters(),
        betas=(args.optimizer.beta1, args.optimizer.beta2),
        eps=args.optimizer.eps,
        weight_decay=args.optimizer.weight_decay)
    
    lr_scheduler = LRScheduler(
        optimizer,
        lr_mul=args.lr_scheduler.lr_mul,
        d_model=args.ranker.d_model,
        n_warmup_steps=args.lr_scheduler.n_warmup_steps)

    sampler = ArchSampler201(
    top_tier=args.sampler.top_tier,
    last_tier= args.sampler.last_tier,
    batch_factor=args.sampler.batch_factor,
    reuse_step=args.sampler.reuse_step,
    )

    assert args.network_type in ['cifar10','cifar100','imagenet16'], 'network type should be one of the [cifar10, cifar100, imagenet16]'
    
    # train ranker
    start = time.perf_counter()
    best_acc = 0
    is_best = False
    for epoch in range(args.start_epochs, args.ranker_epochs):
        flag = args.network_type + ' Ranker Train'
        train_acc, train_loss, distri_list = train_epoch_201(ranker, train_dataloader, criterion, aux_criterion, optimizer, lr_scheduler, device, args, logger, tb_writer, epoch, flag)
        tb_writer.add_scalar('{}/epoch_accuracy'.format(flag), train_acc, epoch)
        tb_writer.add_scalar('{}/epoch_loss'.format(flag), train_loss, epoch)

        # if (epoch+1) % args.validate_freq == 0:
        with torch.no_grad():
            flag = args.network_type + ' Ranker Validate'
            val_acc, val_loss = validate_201(ranker, val_dataloader, criterion, aux_criterion, device, args, logger, epoch, flag)
            tb_writer.add_scalar('{}/epoch_accuracy'.format(flag), val_acc, epoch)
            tb_writer.add_scalar('{}/epoch_loss'.format(flag), val_loss, epoch)

        args.save_path = os.path.join(args.save_dir, 'ckp_last.pth.tar')
        if val_acc > best_acc:
            is_best = True
            best_acc = val_acc
        else:
            is_best = False
        save_checkpoint(args.save_path, ranker, optimizer, lr_scheduler, args, epoch, distri_list, is_best)
    
    logger.info('train ranker using time {:.4f} seconds'.format(time.perf_counter()-start))
    
    # sample
    assert args.sampler_epochs > args.ranker_epochs, 'sampler_epochs should be larger than ranker_epochs'
    assert Bucket.get_n_tier()==0, 'Bucket counts should be reset to 0'
    tier_list = init_tier_list(args)
    
    tpk1_list = []
    tpk5_list = []
    history_best_distri = {}
    tpk1_meter = AverageMeter()
    tpk5_meter = AverageMeter()

    if args.sampler.is_checkpoint:
        ckp_path = os.path.join(args.save_dir, 'ckp_best.pth.tar')
        assert os.path.isfile(ckp_path), 'Checkpoint file does not exist at {}'.format(ckp_path)
        with open(ckp_path, 'rb') as f:
            checkpoint = torch.load(f, map_location=torch.device('cpu'))
        
        distri_list = checkpoint['distri']
        ranker.load_state_dict(checkpoint['state_dict'])
        ranker.cuda(device)

        logger.info('Start to use {} file for sampling'.format(ckp_path))

    random.shuffle(distri_list)
    distri_length = len(distri_list)
    distri_reuse_step = math.ceil((args.sampler_epochs-args.ranker_epochs)/distri_length)
    flag = args.network_type + ' Sample'
    for it in range(args.ranker_epochs, args.sampler_epochs):
        with torch.no_grad():
            if (it-args.ranker_epochs)%distri_reuse_step==0:
                history_best_distri = distri_list[(it-args.ranker_epochs)//distri_reuse_step]

            best_acc_at1, best_rank_at1, best_val_acc_at1, best_acc_at5, best_rank_at5, best_val_acc_at5 = evaluate_sampled_batch_201(ranker, sampler, tier_list, history_best_distri, dataset, it, args, device, tb_writer, logger, flag)
            if best_acc_at1 != None:
                tpk1_meter.update(best_acc_at1, n=1)
                tpk1_list.append((it-args.ranker_epochs, best_acc_at1, best_rank_at1, best_val_acc_at1))
            if best_acc_at5 != None:
                tpk5_meter.update(best_acc_at5, n=1)
                tpk5_list.append((it-args.ranker_epochs, best_acc_at5, best_rank_at5, best_val_acc_at5))
                
    tpk1_best = sorted(tpk1_list, key=lambda item:item[1], reverse=True)[0]
    logger.info('[Result] Top1 Best Arch in Iter {:2d}: Test Acc {:.8f} Val Acc {:.8f} Rank: {:5d}(top {:.2%}), Avg Test Acc {:.8f}'.format(
        tpk1_best[0],
        tpk1_best[1],
        tpk1_best[3],
        tpk1_best[2],
        tpk1_best[2]/len(dataset),
        tpk1_meter.avg))

    tpk5_best = sorted(tpk5_list, key=lambda item:item[1], reverse=True)[0]
    logger.info('[Result] Top5 Best Arch in Iter {:2d}: Test Acc {:.8f} Val Acc {:.8f} Rank: {:5d}(top {:.2%}), Avg Test Acc {:.8f}'.format(
        tpk5_best[0],
        tpk5_best[1],
        tpk5_best[3],
        tpk5_best[2],
        tpk5_best[2]/len(dataset),
        tpk5_meter.avg))

    logger.info('train ranker and search using time {:.4f} seconds'.format(time.perf_counter()-start))

if __name__ == '__main__':
    main()