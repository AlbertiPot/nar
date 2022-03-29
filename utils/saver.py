import os
import torch
import shutil


def save_checkpoint(save_path,
                    model,
                    optimizer,
                    lr_scheduler,
                    args,
                    epoch,
                    distri_list,
                    is_best=False):
    save_state = {
        'epoch': epoch + 1,
        'args': args,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'distri': distri_list,
        'lr_scheduler': lr_scheduler.__dict__
    }

    best_model_path = os.path.join(
        os.path.dirname(save_path),
        'ckp_best.pth.tar')

    with open(save_path, 'wb') as f:
        torch.save(save_state, f, _use_new_zipfile_serialization=False)

    if is_best:
        shutil.copyfile(save_path, best_model_path)