import torch
import torchvision
from tool import imutils

import argparse
import importlib
import numpy as np

import voc12.data
from torch.utils.data import DataLoader
import scipy.misc
import torch.nn.functional as F
import os.path
import copy


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, type=str)
    parser.add_argument("--kernel", default=5, type=int)
    parser.add_argument("--threshold", default=0.02, type=float)
    parser.add_argument("--network", default="network.resnet38_boundary_new", type=str)
    parser.add_argument("--infer_list", default="voc12/train_aug.txt", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--label_dir", default='VOC2012/SegmentationClassAug', type=str)
    parser.add_argument("--voc12_root", default='VOC2012', type=str)
    parser.add_argument("--out_dir", required=True, type=str)
    parser.add_argument("--crf", default=False, type=bool)
    args = parser.parse_args()

    assert (args.kernel % 2 == 1 or args.kernel == 0)
    if args.kernel != 0:
        margin = (args.kernel - 1) // 2

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    
    model = getattr(importlib.import_module(args.network), 'Net')()
    model.load_state_dict(torch.load(args.weights))
    model.eval()
    model.cuda()    

    infer_dataset = voc12.data.VOC12ImageBoundaryDataset(args.infer_list, label_dir=args.label_dir, voc12_root=args.voc12_root,
                    joint_transform_list=[
                        None,
                        None,
                        None
                    ],
                    img_transform_list=[
                        np.asarray,
                        model.normalize,
                        imutils.HWC_to_CHW
                    ],
                    label_transform_list=[
                        None,
                        None,
                        None
                    ])

    infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    for iter, (img_name, img, label) in enumerate(infer_data_loader):

        print(args.out_dir, iter)
        name = img_name[0]

        with torch.no_grad():

            label1 = model(img.cuda())
            prediction = torch.sigmoid(label1).detach() # (1,1,H,W), torch.float

            ones = torch.ones(prediction.shape).cuda()
            zeros = torch.zeros(prediction.shape).cuda()
            nonboundary_region = torch.where(prediction < args.threshold, ones, zeros)
            if args.kernel != 0:
                nonboundary_region_padded = F.pad(nonboundary_region, (margin, margin, margin, margin), "replicate")
                nonboundary_region_avg = F.avg_pool2d(nonboundary_region_padded, kernel_size=(args.kernel,args.kernel), stride=1)
                target_pixel = torch.where(nonboundary_region_avg >= 0.99, ones, zeros)
            else:
                target_pixel = nonboundary_region
            target_pixel = F.interpolate(target_pixel.float(), size=img.shape[2:], mode="nearest")        
            target_res = np.uint8(target_pixel.long().cpu().data[0,0] * 255)
            scipy.misc.imsave(os.path.join(args.out_dir, name + '.png'), target_res)

