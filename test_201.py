import os
import math
import time
import torch
import random
import argparse
import torch.utils.data

from dataset import NASBench201DataBase, NASBench201Dataset, SplitSubet201
from architecture import Bucket
from ranker import Transformer
from sampler import ArchSampler201
from utils.loss_ops import CrossEntropyLossSoft, RankLoss
from utils.metric import AverageMeter
from utils.setup import setup_seed, setup_logger
from utils.config import get_config
from process import validate_201, evaluate_sampled_batch_201
from process.train_utils import init_tier_list


def get_args():
    parser = argparse.ArgumentParser(description='NAR Testing for nasbench201')
    parser.add_argument('--config_file',
                        default='./config/config201.yml',
                        type=str,
                        help='testing configuration')
    parser.add_argument('--data_path',
                        default='./data/nasbench201/nasbench201_with_edge_flops_and_params.json',
                        type=str,
                        help='Path to load data')
    parser.add_argument('--save_dir',
                        default='./output/fixed_labels/n201_img16_noisy0_baseline/n201_img16_noisy0_exp_seed99999999_20211105150803',
                        type=str,
                        help='Path to save output')
    parser.add_argument('--checkpoint',
                        default='ckp_best.pth.tar',
                        type=str,
                        help='checkpoint file')
    parser.add_argument('--save_file_name',
                        default='noisy05_test.log',
                        type=str,
                        help='save file name')

    args = parser.parse_args()

    return args

def main():
    run_args = get_args()
    
    args = get_config(run_args.config_file)
    args.config_file = run_args.config_file
    args.data_path = run_args.data_path
    args.save_dir = run_args.save_dir
    args.save_file_name = run_args.save_file_name
    
    ckp_path = os.path.join(run_args.save_dir, run_args.checkpoint)
    assert os.path.isfile(ckp_path), 'Checkpoint file does not exist at {}'.format(ckp_path)
    with open(ckp_path, 'rb') as f:
        checkpoint = torch.load(f, map_location=torch.device('cpu'))

    logger = setup_logger(save_path=os.path.join(args.save_dir, args.save_file_name))
    logger.info(args)

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
        valset = SplitSubet201(dataset, list(range(args.train_size, args.train_size + args.val_size)), args.ranker.n_tier)
        
    # build dataloader
    val_dataloader = torch.utils.data.DataLoader(
        valset,
        batch_size=args.val_batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.data_loader_workers,
        pin_memory=True)

    criterion = CrossEntropyLossSoft().cuda(device)
    if args.aux_loss:
        aux_criterion = RankLoss().cuda(device)

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
    
    ranker.load_state_dict(checkpoint['state_dict'])
    ranker.cuda(device)

    sampler = ArchSampler201(
    top_tier=args.sampler.top_tier,
    last_tier= args.sampler.last_tier,
    batch_factor=args.sampler.batch_factor,
    reuse_step=args.sampler.reuse_step,
    )

    assert args.network_type in ['cifar10','cifar100','imagenet16'], 'network type should be one of the [cifar10, cifar100, imagenet16]'

    logger.info('Start to use {} file for testing ranker and sampling'.format(ckp_path))
    with torch.no_grad():
        flag = args.network_type + ' Ranker Test'
        val_acc, val_loss = validate_201(ranker, val_dataloader, criterion, aux_criterion, device, args, logger, 0, flag)

    # sample
    start = time.perf_counter()
    assert args.sampler_epochs > args.ranker_epochs, 'sampler_epochs should be larger than ranker_epochs'
    assert Bucket.get_n_tier()==0, 'Bucket counts should be reset to 0'
    tier_list = init_tier_list(args)
    
    tpk1_list = []
    tpk5_list = []
    history_best_distri = {}
    tpk1_meter = AverageMeter()
    tpk5_meter = AverageMeter()

    distri_list = checkpoint['distri']
    random.shuffle(distri_list)
    distri_length = len(distri_list)
    distri_reuse_step = math.ceil((args.sampler_epochs-args.ranker_epochs)/distri_length)
    flag = args.network_type + ' Sample Test'
    for it in range(args.ranker_epochs, args.sampler_epochs):
        with torch.no_grad():
            if (it-args.ranker_epochs)%distri_reuse_step==0:
                history_best_distri = distri_list[(it-args.ranker_epochs)//distri_reuse_step]

            best_acc_at1, best_rank_at1, best_val_acc_at1, best_acc_at5, best_rank_at5, best_val_acc_at5 = evaluate_sampled_batch_201(ranker, sampler, tier_list, history_best_distri, dataset, it, args, device, None, logger, flag)
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
    
    logger.info('search using time {:.4f} seconds'.format(time.perf_counter()-start))

if __name__ == '__main__':
        main()