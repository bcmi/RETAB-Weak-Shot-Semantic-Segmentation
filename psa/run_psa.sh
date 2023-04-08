#!/bin/bash

mkdir result
mkdir result/psa_trainaug_cam
mkdir result/psa_trainaug_crf_4.0
mkdir result/psa_trainaug_crf_32.0
python infer_cls.py --infer_list voc12/train_aug.txt --voc12_root VOC2012 --network network.resnet38_cls --weights best/res38_cls.pth --out_cam result/psa_trainaug_cam --out_la_crf result/psa_trainaug_crf_4.0 --out_ha_crf result/psa_trainaug_crf_32.0
python evaluation.py --logfile evalog_psa.txt --list VOC2012/ImageSets/Segmentation/train.txt --predict_dir result/psa_trainaug_cam --gt_dir VOC2012/SegmentationClassAug --comment psa_train_cam --type npy --curve True
