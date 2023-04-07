
import numpy as np
import torch
import random
import cv2
import os
from torch.utils.data import DataLoader
from torchvision import transforms
import voc12.data
from tool import pyutils, imutils, torchutils, visualization
import argparse
import importlib
from tensorboardX import SummaryWriter
import torch.nn.functional as F


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--max_epoches", default=8, type=int)
    parser.add_argument("--network", default="network.resnet38_boundary_new", type=str)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument("--train_list", required=True, type=str)
    parser.add_argument("--val_list", default="voc12/val.txt", type=str)
    parser.add_argument("--session_name", required=True, type=str)
    parser.add_argument("--crop_size", default=448, type=int)
    parser.add_argument("--weights", required=True, type=str)
    parser.add_argument("--label_dir", default='VOC2012/SegmentationClassAug', type=str)
    parser.add_argument("--voc12_root", default='VOC2012', type=str)
    parser.add_argument("--tblog_dir", default='./tblog', type=str)
    args = parser.parse_args()

    pyutils.Logger(args.session_name + '.log')

    print(vars(args))
    
    model = getattr(importlib.import_module(args.network), 'Net')()

    print(model)

    if not os.path.exists(args.tblog_dir):
        os.makedirs(args.tblog_dir)
    tblogger = SummaryWriter(args.tblog_dir)	

    train_dataset = voc12.data.VOC12ImageBoundaryDatasetNew(args.train_list, label_dir=args.label_dir, voc12_root=args.voc12_root,
                    joint_transform_list=[
                        None,
                        None,
                        imutils.RandomCrop(args.crop_size),
                        imutils.RandomHorizontalFlip()
                    ],
                    img_transform_list=[
                        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                        np.asarray,
                        model.normalize,
                        imutils.HWC_to_CHW
                    ],
                    label_transform_list=[
                        None,
                        None,
                        None,
                        None #imutils.AvgPool2d(8)
                    ])
    def worker_init_fn(worker_id):
        np.random.seed(1 + worker_id)
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True,
                                   worker_init_fn=worker_init_fn)
    max_step = len(train_dataset) // args.batch_size * args.max_epoches

    param_groups = model.get_parameter_groups()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[1], 'lr': 2*args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10*args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[3], 'lr': 20*args.lr, 'weight_decay': 0}
    ], lr=args.lr, weight_decay=args.wt_dec, max_step=max_step)

    if args.weights[-7:] == '.params':
        import network.resnet38d
        assert 'resnet38' in args.network
        weights_dict = network.resnet38d.convert_mxnet_to_torch(args.weights)
    else:
        weights_dict = torch.load(args.weights)

    model.load_state_dict(weights_dict, strict=False)
    model = torch.nn.DataParallel(model).cuda()
    model.train()

    avg_meter = pyutils.AverageMeter('loss')
    criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

    timer = pyutils.Timer("Session started: ")
    for ep in range(args.max_epoches):

        for iter, pack in enumerate(train_data_loader):

            img1 = pack[1]
            prediction = model(img1)

            seg_label = pack[2].cuda(non_blocking=True)
            #seg_label = F.interpolate(seg_label.unsqueeze(1).float(), prediction.size()[2:], mode='nearest').squeeze(1).long()
            ones = torch.ones(seg_label.shape).cuda(non_blocking=True)
            zeros = torch.zeros(seg_label.shape).cuda(non_blocking=True)
            bry_mask = torch.where(seg_label==255, ones, zeros).float()
            bg_mask = torch.where(seg_label==0, ones, zeros).float()
            fg_mask = 1.0 - (bry_mask + bg_mask)

            loss_all = criterion(prediction.squeeze(1), bry_mask)
            loss = 0.0
            if bry_mask.sum() > 0.5:
                loss += (loss_all * bry_mask).sum() / bry_mask.sum()
            if bg_mask.sum() > 0.5:
                loss += 0.5 * (loss_all * bg_mask).sum() / bg_mask.sum()
            if fg_mask.sum() > 0.5:
                loss += 0.5 * (loss_all * fg_mask).sum() / fg_mask.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_meter.add({'loss': loss.item()})

            if (optimizer.global_step - 1) % 10 == 0:

                timer.update_progress(optimizer.global_step / max_step)

                print('Iter:%5d/%5d' % (optimizer.global_step-1, max_step),
                      'loss:%.4f' % avg_meter.get('loss'),
                      'imps:%.1f' % ((iter+1) * args.batch_size / timer.get_stage_elapsed()),
                      'Fin:%s' % (timer.str_est_finish()),
                      'lr: %.5f' % (optimizer.param_groups[0]['lr']), flush=True)

                avg_meter.pop()

                # Visualization for training process
                # img_8 = img1[0].numpy().transpose((1,2,0))
                # img_8 = np.ascontiguousarray(img_8)
                # mean = (0.485, 0.456, 0.406)
                # std = (0.229, 0.224, 0.225)
                # img_8[:,:,0] = (img_8[:,:,0]*std[0] + mean[0])*255
                # img_8[:,:,1] = (img_8[:,:,1]*std[1] + mean[1])*255
                # img_8[:,:,2] = (img_8[:,:,2]*std[2] + mean[2])*255
                # img_8[img_8 > 255] = 255
                # img_8[img_8 < 0] = 0
                # img_8 = img_8.astype(np.uint8)

                # input_img = img_8.transpose((2,0,1))
                # h = H//4; w = W//4
                # p1 = F.interpolate(cam1,(h,w),mode='bilinear')[0].detach().cpu().numpy()
                # p2 = F.interpolate(cam2,(h,w),mode='bilinear')[0].detach().cpu().numpy()
                # p_rv1 = F.interpolate(cam_rv1,(h,w),mode='bilinear')[0].detach().cpu().numpy()
                # p_rv2 = F.interpolate(cam_rv2,(h,w),mode='bilinear')[0].detach().cpu().numpy()

                # image = cv2.resize(img_8, (w,h), interpolation=cv2.INTER_CUBIC).transpose((2,0,1))
                # CLS1, CAM1, _, _ = visualization.generate_vis(p1, None, image, func_label2color=visualization.VOClabel2colormap, threshold=None, norm=False)
                # CLS2, CAM2, _, _ = visualization.generate_vis(p2, None, image, func_label2color=visualization.VOClabel2colormap, threshold=None, norm=False)
                # CLS_RV1, CAM_RV1, _, _ = visualization.generate_vis(p_rv1, None, image, func_label2color=visualization.VOClabel2colormap, threshold=None, norm=False)
                # CLS_RV2, CAM_RV2, _, _ = visualization.generate_vis(p_rv2, None, image, func_label2color=visualization.VOClabel2colormap, threshold=None, norm=False)
                # #MASK = eq_mask[0].detach().cpu().numpy().astype(np.uint8)*255
                # loss_dict = {'loss':loss.item(), 
                #              'loss_cls':loss_cls.item(),
                #              'loss_er':loss_er.item(),
                #              'loss_ecr':loss_ecr.item()}
                # itr = optimizer.global_step - 1
                # tblogger.add_scalars('loss', loss_dict, itr)
                # tblogger.add_scalar('lr', optimizer.param_groups[0]['lr'], itr)
                # tblogger.add_image('Image', input_img, itr)
                # #tblogger.add_image('Mask', MASK, itr)
                # tblogger.add_image('CLS1', CLS1, itr)
                # tblogger.add_image('CLS2', CLS2, itr)
                # tblogger.add_image('CLS_RV1', CLS_RV1, itr)
                # tblogger.add_image('CLS_RV2', CLS_RV2, itr)
                # tblogger.add_images('CAM1', CAM1, itr)
                # tblogger.add_images('CAM2', CAM2, itr)
                # tblogger.add_images('CAM_RV1', CAM_RV1, itr)
                # tblogger.add_images('CAM_RV2', CAM_RV2, itr)

        else:
            print('')
            timer.reset_stage()

    torch.save(model.module.state_dict(), args.session_name + '.pth')

         
