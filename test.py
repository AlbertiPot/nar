import os
import math
import time
import torch
import random
import argparse
import torch.utils.data

from dataset import NASBenchDataBase, NASBenchDataset, SplitSubet
from architecture import Bucket
from ranker import Transformer
from sampler import ArchSampler
from utils.metric import AverageMeter
from utils.loss_ops import CrossEntropyLossSoft, RankLoss
from utils.config import get_config
from utils.setup import setup_seed, setup_logger
from process import validate, evaluate_sampled_batch
from process.train_utils import init_tier_list


def get_args():
    parser = argparse.ArgumentParser(description='NAR Testing for nasbench101')
    parser.add_argument('--config_file',
                        default='./config/config.yml',
                        type=str,
                        help='testing configuration')
    parser.add_argument('--data_path',
                        default='./data/nasbench101/nasbench_only108_with_vertex_flops_and_params.json',
                        type=str,
                        help='Path to load data')
    parser.add_argument('--save_dir',
                        default='./output/fixed_labels/n101_noisy0_baseline/n101_noisy0_seed77777777_20211105102057',
                        type=str,
                        help='Path to save output')
    parser.add_argument('--checkpoint',
                        default='ckp_best.pth.tar',
                        type=str,
                        help='checkpoint file')
    parser.add_argument('--seed',
                        default=77777777,
                        type=int,
                        help='set seed')                    
    parser.add_argument('--save_file_name',
                        default='tttt.log',
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
    args.seed = run_args.seed
    
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
    if args.space == 'nasbench':
        database = NASBenchDataBase(args.data_path)
        dataset = NASBenchDataset(database, seed=args.seed)
        valset = SplitSubet(dataset, list(range(args.train_size, args.train_size + args.val_size)), args.ranker.n_tier)
        
    # build dataloader
    val_dataloader = torch.utils.data.DataLoader(
        valset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.data_loader_workers,
        pin_memory=True)

    criterion = CrossEntropyLossSoft().cuda(device)
    if args.aux_loss:
        aux_criterion = RankLoss().cuda(device)
    else:
        aux_criterion = None

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

    sampler = ArchSampler(
    top_tier=args.sampler.top_tier,
    last_tier= args.sampler.last_tier,
    batch_factor=args.sampler.batch_factor,
    node_type_dict=dict(args.node_type_dict),
    max_edges=args.max_edges,
    reuse_step=args.sampler.reuse_step,
    )

    logger.info('Start to use {} file for testing ranker and sampling'.format(ckp_path))
    with torch.no_grad():
        flag = 'Ranker Test'
        val_acc, val_loss = validate(ranker, val_dataloader, criterion, aux_criterion, device, args, logger, 0, flag)

    # sample
    start = time.perf_counter()
    assert args.sampler_epochs > args.ranker_epochs, 'sampler_epochs should be larger than ranker_epochs'
    assert Bucket.get_n_tier()==0, 'Bucket counts should be reset to 0'
    tier_list = init_tier_list(args)
    
    history_best_distri = {}
    tpk1_list = []
    tpk5_list = []
    tpk3_list = []
    tpk7_list = []
    tpk10_list = []

    tpk1_meter = AverageMeter()
    tpk5_meter = AverageMeter()
    tpk3_meter = AverageMeter()
    tpk7_meter = AverageMeter()
    tpk10_meter = AverageMeter()

    distri_list = checkpoint['distri']
    random.shuffle(distri_list)
    distri_length = len(distri_list)
    distri_reuse_step = math.ceil((args.sampler_epochs-args.ranker_epochs)/distri_length)
    flag = 'Sample Test'
    for it in range(args.ranker_epochs, args.sampler_epochs):
        with torch.no_grad():
            if (it-args.ranker_epochs)%distri_reuse_step==0:
                history_best_distri = distri_list[(it-args.ranker_epochs)//distri_reuse_step]

            best_acc_at1, best_rank_at1, best_acc_at5, best_rank_at5, best_acc_at3, best_rank_at3, best_acc_at7, best_rank_at7, best_acc_at10, best_rank_at10 = evaluate_sampled_batch(ranker, sampler, tier_list, history_best_distri, dataset, it, args, device, None, logger, flag)
            
            tpk1_meter.update(best_acc_at1, n=1)
            tpk1_list.append((it-args.ranker_epochs, best_acc_at1, best_rank_at1))
            tpk5_meter.update(best_acc_at5, n=1)
            tpk5_list.append((it-args.ranker_epochs, best_acc_at5, best_rank_at5))

            tpk3_meter.update(best_acc_at3, n=1)
            tpk3_list.append((it-args.ranker_epochs, best_acc_at3, best_rank_at3))
            tpk7_meter.update(best_acc_at7, n=1)
            tpk7_list.append((it-args.ranker_epochs, best_acc_at7, best_rank_at7))
            tpk10_meter.update(best_acc_at10, n=1)
            tpk10_list.append((it-args.ranker_epochs, best_acc_at10, best_rank_at10))
    
    tpk1_best = sorted(tpk1_list, key=lambda item:item[1], reverse=True)[0]
    logger.info('[Result] Top1 Best Arch in Iter {:2d}: Test Acc {:.8f} Rank: {:5d}(top {:.2%}), Avg Test Acc {:.8f}'.format(
        tpk1_best[0],
        tpk1_best[1],  
        tpk1_best[2],
        tpk1_best[2]/len(dataset),
        tpk1_meter.avg))

    tpk5_best = sorted(tpk5_list, key=lambda item:item[1], reverse=True)[0]
    logger.info('[Result] Top5 Best Arch in Iter {:2d}: Test Acc {:.8f} Rank: {:5d}(top {:.2%}), Avg Test Acc {:.8f}'.format(
        tpk5_best[0],
        tpk5_best[1],  
        tpk5_best[2],
        tpk5_best[2]/len(dataset),
        tpk5_meter.avg))

    tpk3_best = sorted(tpk3_list, key=lambda item:item[1], reverse=True)[0]
    logger.info('[Result] Top3 Best Arch in Iter {:2d}: Test Acc {:.8f} Rank: {:5d}(top {:.2%}), Avg Test Acc {:.8f}'.format(
        tpk3_best[0],
        tpk3_best[1],  
        tpk3_best[2],
        tpk3_best[2]/len(dataset),
        tpk3_meter.avg))

    tpk7_best = sorted(tpk7_list, key=lambda item:item[1], reverse=True)[0]
    logger.info('[Result] Top7 Best Arch in Iter {:2d}: Test Acc {:.8f} Rank: {:5d}(top {:.2%}), Avg Test Acc {:.8f}'.format(
        tpk7_best[0],
        tpk7_best[1],  
        tpk7_best[2],
        tpk7_best[2]/len(dataset),
        tpk7_meter.avg))
    
    tpk10_best = sorted(tpk10_list, key=lambda item:item[1], reverse=True)[0]
    logger.info('[Result] Top10 Best Arch in Iter {:2d}: Test Acc {:.8f} Rank: {:5d}(top {:.2%}), Avg Test Acc {:.8f}'.format(
        tpk10_best[0],
        tpk10_best[1],  
        tpk10_best[2],
        tpk10_best[2]/len(dataset),
        tpk10_meter.avg))
    
    logger.info('search using time {:.4f} seconds'.format(time.perf_counter()-start))

if __name__ == '__main__':
        main()


