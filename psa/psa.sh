#!/bin/bash

# mkdir result/psa_val_cam
# mkdir result/psa_val_crf_4.0
# mkdir result/psa_val_crf_32.0
# python infer_cls.py --infer_list voc12/val.txt --voc12_root VOC2012 --network network.resnet38_cls --weights best/res38_cls.pth --out_cam result/psa_val_cam --out_la_crf result/psa_val_crf_4.0 --out_ha_crf result/psa_val_crf_32.0
# python evaluation.py --logfile evalog_psa.txt --list VOC2012/ImageSets/Segmentation/val.txt --predict_dir result/psa_val_cam --gt_dir VOC2012/SegmentationClassAug --comment psa_val_cam --type npy --curve True


mkdir result/psa_trainaug_cam
mkdir result/psa_trainaug_crf_4.0
mkdir result/psa_trainaug_crf_32.0
python infer_cls.py --infer_list voc12/train_aug.txt --voc12_root VOC2012 --network network.resnet38_cls --weights best/res38_cls.pth --out_cam result/psa_trainaug_cam --out_la_crf result/psa_trainaug_crf_4.0 --out_ha_crf result/psa_trainaug_crf_32.0
python evaluation.py --logfile evalog_psa.txt --list VOC2012/ImageSets/Segmentation/train.txt --predict_dir result/psa_trainaug_cam --gt_dir VOC2012/SegmentationClassAug --comment psa_train_cam --type npy --curve True
# python evaluation_pngnew.py --logfile evalog_psa.txt --list VOC2012/ImageSets/Segmentation/train.txt --predict_dir result/psa_trainaug_cam --gt_dir VOC2012/SegmentationClassAug --comment psa_train_cam --type npy --curve True
# python evaluation.py --logfile evalog_psa.txt --list VOC2012/ImageSets/SegmentationAug/train_aug_new.txt --predict_dir result/psa_trainaug_cam --gt_dir VOC2012/SegmentationClassAug --comment psa_trainaug_cam --type npy --curve True


# a=16
# b=8
# c=8
# mkdir result/psa_trainaug_affrw
# mkdir result/psa_trainaug_affrw_cam
# python infer_aff_cam.py --alpha ${a} --beta ${b} --logt ${c} --infer_list voc12/train_aug.txt --voc12_root VOC2012 --network network.resnet38_aff --weights best/res38_aff.pth --cam_dir result/psa_trainaug_cam --out_rw result/psa_trainaug_affrw --out_rw_cam result/psa_trainaug_affrw_cam
# python evaluation.py --logfile evalog_psa.txt --list VOC2012/ImageSets/Segmentation/train.txt --predict_dir result/psa_trainaug_affrw --gt_dir VOC2012/SegmentationClassAug --comment psa_res38_train_affrw --type png
