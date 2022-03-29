import torch
import torch.nn.functional as F


def compute_accuracy(output, target):
    batch_sz = target.size(0)  # both target and output are b_sz * n_class
    _, pred = output.topk(k=1, dim=1)
    _, label = target.topk(k=1, dim=1)
    correct = pred.eq(label).sum()

    return float(torch.true_divide(correct, batch_sz))


def _sign(number):
    if isinstance(number, (list, tuple)):
        return [_sign(v) for v in number]
    if number >= 0.0:
        return 1
    elif number < 0.0:
        return -1


def compute_kendall_tau(pred_score, score):
    '''
    Kendall Tau is a metric to measure the ordinal association between two measured quantities.
    Refer to https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient
    '''

    pred_score = pred_score.squeeze(dim=-1)
    assert len(pred_score) == len(score), "Sequence a and b should have the same length while computing kendall tau."
    
    length = len(pred_score)
    count = 0
    total = 0
    for i in range(length - 1):
        for j in range(i + 1, length):
            count += _sign(pred_score[i] - pred_score[j]) * _sign(score[i] - score[j])
            total += 1
    Ktau = count / total
    return Ktau


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count