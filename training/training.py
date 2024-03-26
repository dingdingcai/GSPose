import os
gpu_id = 0
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

import sys
import time
import glob
import torch
import shutil
import numpy as np
from torch import optim
import matplotlib.pyplot as plt

from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # ./../../
sys.path.append(PROJ_ROOT)

from dataset import misc
from misc_utils import warmup_lr
from model.network import model_arch as ModelNet
from dataset.megapose_dataset import MegaPose_Dataset as Dataset
device = torch.device('cuda:0')

def batchify_cuda_device(data_dict, batch_size, flatten_multiview=True, use_cuda=True):
    for key, val in data_dict.items():
        for sub_key, sub_val in val.items():
            if use_cuda:
                try:
                    data_dict[key][sub_key] = sub_val.cuda(non_blocking=True)
                except:
                    pass
            if flatten_multiview:
                try:
                    if data_dict[key][sub_key].shape[0] == batch_size:
                        data_dict[key][sub_key] = data_dict[key][sub_key].flatten(0, 1)
                except:
                    pass

img_size = 224
batch_size = 2
que_view_num = 4
refer_view_num = 8
random_view_num = 24    # 8 + 24 = 32
nnb_Rmat_threshold = 30
num_train_iters = 100_000

DATA_DIR = os.path.join(PROJ_ROOT, 'dataspace', 'MegaPose')

dataset = Dataset(data_dir=DATA_DIR,
                  query_view_num=que_view_num,
                  refer_view_num=refer_view_num,
                  rand_view_num=random_view_num,
                  nnb_Rmat_threshold=nnb_Rmat_threshold, 
                 )

print('num_objects: ', len(dataset.selected_objIDs))

model_net = ModelNet().to(device)
CKPT_ROOT = os.path.join(PROJ_ROOT, 'checkpoints')
checkpoints = os.path.join(CKPT_ROOT, 'checkpoints')
tb_dir = os.path.join(checkpoints, 'tb')
tb_old = tb_dir.replace('tb', 'tb_old')
if os.path.exists(tb_old):
    shutil.rmtree(tb_old)
if not os.path.exists(tb_dir):
    os.makedirs(tb_dir)
shutil.move(tb_dir, tb_old)
tb_writer = SummaryWriter(tb_dir)

data_loader = torch.utils.data.DataLoader(dataset,
                                            shuffle=True,
                                            num_workers=8, 
                                            batch_size=batch_size, 
                                            collate_fn=dataset.collate_fn,
                                            pin_memory=False, drop_last=False)

END_LR = 1e-6
START_LR = 1e-4
max_steps = num_train_iters

iter_steps = 0
TB_SKIP_STEPS = 5
short_log_interval = 100
long_log_interval = 1_000
checkpoint_interval = 10_000
enable_FP16_training = True
LOSS_WEIGHTS = {
    'rm_loss': 1.0,
    'cm_loss': 10.0,
    'qm_loss': 10.0,
    'Remb_loss': 1.0,
}

optimizer = optim.AdamW(model_net.parameters(), lr=START_LR)
lr_scheduler = warmup_lr.CosineAnnealingWarmupRestarts(optimizer, max_steps, max_lr=START_LR, min_lr=END_LR)

losses_dict = {}
model_net.train()
scaler = GradScaler()
start_timer = time.time()
data_iterator = iter(data_loader)

print('total training max_steps: {}'.format(max_steps))
print('enable_FP16_training: ', enable_FP16_training)
for iter_steps in range(1, max_steps+1):
    lr_scheduler.step()
    optimizer.zero_grad()
    try:
        batch_data = next(data_iterator)
    except:
        data_iterator = iter(data_loader) # reinitialize the iterator
        batch_data = next(data_iterator)

    batchify_cuda_device(batch_data, batch_size=batch_size, flatten_multiview=True, use_cuda=True)        
    scaler_curr_scale = 1.0
    loss = 0
    with autocast(enable_FP16_training):
        net_outputs = model_net(batch_data)
        for ls_name, ls_wgh in LOSS_WEIGHTS.items():
            loss += net_outputs.get(ls_name, 0) * LOSS_WEIGHTS[ls_name]
            assert (not torch.isnan(loss).any())

    scaler.scale(loss).backward()

    with torch.no_grad():
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model_net.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scaler_curr_scale = scaler.state_dict()['scale']

        if 'ls' not in losses_dict:
            losses_dict['ls'] = list()
        losses_dict['ls'].append(loss.item())

        for key_, val_ in net_outputs.items():
            if 'loss' in key_:
                if key_ not in losses_dict:
                    losses_dict[key_] = list()
                ls = val_.item()
                if key_ in LOSS_WEIGHTS:
                    ls *= LOSS_WEIGHTS[key_]
                losses_dict[key_].append(ls) 

        if (iter_steps > TB_SKIP_STEPS) and (iter_steps % short_log_interval == 0):
            tb_writer.add_scalar("Other/lr", optimizer.param_groups[0]['lr'], iter_steps)

            for idx, (key_, val_) in enumerate(losses_dict.items()):
                tb_writer.add_scalar(f"Loss/{idx}_{key_}", val_[-1], iter_steps)

        if ((iter_steps > 5 and iter_steps < 2000 and iter_steps % short_log_interval == 0)
            or iter_steps % long_log_interval == 0):

            curr_lr = optimizer.param_groups[0]['lr']
            time_stamp = time.strftime('%d-%H:%M:%S', time.localtime())
            logging_str = "{:.1f}k".format(iter_steps/1000)

            for key_, val_ in losses_dict.items():
                dis_str = key_.split('_')[0]
                logging_str += ', {}:{:.4f}'.format(dis_str, np.mean(val_[-2000:]))

            logging_str += ', {}'.format(time_stamp)
            logging_str += ', {:.1f}'.format(scaler_curr_scale)
            logging_str += ', {:.6f}'.format(curr_lr)
            
            print(logging_str)

            vis_num_views = np.minimum(8, refer_view_num)
            fig, ax = plt.subplots(4, vis_num_views+1, figsize=(12, 5),
                    gridspec_kw={'width_ratios': [1.5] + [1 for _ in range(vis_num_views)]}
            )

            rgb_que_image = batch_data['query_dict']['rescaled_image'][0].detach().cpu().permute(1, 2, 0).squeeze().float()
            gt_que_full_mask = batch_data['query_dict']['rescaled_mask'][0].detach().cpu().permute(1, 2, 0).squeeze().float()
            pd_que_full_mask = net_outputs['que_full_pd_mask'][0].detach().cpu().permute(1, 2, 0).squeeze().float()
            rgb_path = batch_data['query_dict']['rgb_path'][0].split('train_pbr/')[-1]
            ax[0, 0].imshow(rgb_que_image)
            ax[0, 0].set_title(rgb_path, fontsize=10)
            ax[1, 0].imshow(gt_que_full_mask)
            ax[2, 0].imshow(pd_que_full_mask)
            ax[3, 0].imshow((gt_que_full_mask - pd_que_full_mask))
            ax[0, 0].axis(False)
            ax[1, 0].axis(False)
            ax[2, 0].axis(False)
            ax[3, 0].axis(False)

            rgb_que_image = batch_data['query_dict']['dzi_image'][0].detach().cpu().permute(1, 2, 0).squeeze().float()
            gt_que_que_mask = batch_data['query_dict']['dzi_mask'][0].detach().cpu().permute(1, 2, 0).squeeze().float()
            pd_que_que_mask = net_outputs['que_pd_mask'][0].detach().cpu().permute(1, 2, 0).squeeze().float()
            ax[0, 1].imshow(rgb_que_image)
            ax[1, 1].imshow(gt_que_que_mask)
            ax[2, 1].imshow(pd_que_que_mask)
            ax[3, 1].imshow((gt_que_que_mask - pd_que_que_mask))
            ax[0, 1].axis(False)
            ax[1, 1].axis(False)
            ax[2, 1].axis(False)
            ax[3, 1].axis(False)

            for vix in range(vis_num_views-1):
                vjx = vix + 2
                rgb_ref_image = batch_data['refer_dict']['zoom_image'][vix].detach().cpu().permute(1, 2, 0).squeeze().float()
                gt_ref_mask = batch_data['refer_dict']['zoom_mask'][vix].detach().cpu().permute(1, 2, 0).squeeze().float()
                pd_ref_mask = net_outputs['ref_pd_mask'][vix].detach().cpu().permute(1, 2, 0).squeeze().float()
                ax[0, vjx].imshow(rgb_ref_image)
                ax[1, vjx].imshow(gt_ref_mask)
                ax[2, vjx].imshow(pd_ref_mask)
                ax[3, vjx].imshow((gt_ref_mask - pd_ref_mask))
                ax[0, vjx].axis(False)
                ax[1, vjx].axis(False)
                ax[2, vjx].axis(False)
                ax[3, vjx].axis(False)
            plt.tight_layout()
            tb_writer.add_figure('visulize_refer', fig, iter_steps)
            fig.clear()

            Remb_logit = net_outputs['Remb_logit'][0].detach().cpu()
            delta_Rdeg = net_outputs['delta_Rdeg'][0].detach().cpu().float()
            delta_Rdeg = torch.acos(torch.clamp(delta_Rdeg, min=-1.0, max=1.0)) / torch.pi * 180    
            rank_Rdegs, rank_Rinds = torch.topk(delta_Rdeg, dim=0, k=delta_Rdeg.shape[0], largest=False)

            fig, ax = plt.subplots(1, 1)
            ax.plot(rank_Rdegs, Remb_logit[rank_Rinds])
            ax.grid()           
            tb_writer.add_figure('Rotation probability distribution', fig, iter_steps)
            fig.clear()
        
        if iter_steps % checkpoint_interval == 0:
            if not os.path.exists(checkpoints):
                os.makedirs(checkpoints)
            time_stamp = time.strftime('%m%d_%H%M%S', time.localtime())

            ckpt_name = 'model_{}_{}.pth'.format(iter_steps, time_stamp)
            ckpt_file = os.path.join(checkpoints, ckpt_name) 
            try:
               torch.save(model_net.module.state_dict(), ckpt_file)
            except:
               torch.save(model_net.state_dict(), ckpt_file)
            
            # try:
            #     state = {
            #         'model_net': model_net.module.state_dict(),
            #         'optimizer': optimizer.state_dict(),
            #         'lr_scheduler': lr_scheduler.state_dict(),
            #         'scaler': scaler.state_dict(),
            #         'iter_steps': iter_steps,
            #         }
            #     torch.save(state, ckpt_file)
            # except:
            #     state = {
            #         'model_net': model_net.state_dict(),
            #         'optimizer': optimizer.state_dict(),
            #         'lr_scheduler': lr_scheduler.state_dict(),
            #         'scaler': scaler.state_dict(),
            #         'iter_steps': iter_steps,
            #         }
            #     torch.save(state, ckpt_file)
            
            print('saving to ', ckpt_file)

 
