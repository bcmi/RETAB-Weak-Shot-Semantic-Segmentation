#!/bin/bash

## Pascal-5i category split, FOLD is i=0,1,2,3
FOLD=0

## experiment name
EXPID=ourbest

# hyper-parameters
KERNEL=0
THRESHOLD=0.5

## pretrained model path
PRETRAINDIR=pretrained_model

## trained boundary weights and output boundary prediction
BOUNDARYWEIGHTDIR=boundary_model
mkdir ${BOUNDARYWEIGHTDIR}
BOUNDARYPREDDIR=boundary_pred
mkdir ${BOUNDARYPREDDIR}

## affinitynet training and model saving
AFFMETHOD=psa
INITTRAINDIR=${AFFMETHOD}_afflabel
INITCAMCRFLA=psa_trainaug_crf_4.0
INITCAMCRFHA=psa_trainaug_crf_32.0
AFFWEIGHTDIR=${AFFMETHOD}_affmodel
mkdir ${AFFWEIGHTDIR}

## affinitynet inferring and random walk saving
INFERMETHOD=psa
INITCAMDIR=${INFERMETHOD}_initcam
INITCAMTRAINAUG=psa_trainaug_cam
INITCAMVAL=psa_val_cam
OUTRWDIR=${INFERMETHOD}_affrw
mkdir ${OUTRWDIR}

## evaluation dir
mkdir evallog_${INFERMETHOD}

## train the boundary network
python train_boundary.py\
  --train_list voc12/trainaug_fold${FOLD}_base.txt\
  --weights ${PRETRAINDIR}/ilsvrc-cls_rna-a1_cls1000_ep-0001.params\
  --tblog_dir ./tblog/resnet38_boundary_fold${FOLD}\
  --session_name ${BOUNDARYWEIGHTDIR}/resnet38_boundary_fold${FOLD}

## predict the boundary map
python prop_boundary_target.py\
  --infer_list voc12/train_aug.txt\
  --weights ${BOUNDARYWEIGHTDIR}/resnet38_boundary_fold${FOLD}.pth\
  --kernel ${KERNEL}\
  --threshold ${THRESHOLD}\
  --out_dir ${BOUNDARYPREDDIR}/boundary_fold${FOLD}_kernel${KERNEL}_threshold${THRESHOLD}

## train the affinity network
python train_affmixed_boundary.py\
  --train_list voc12/train_aug.txt\
  --weights ${PRETRAINDIR}/ilsvrc-cls_rna-a1_cls1000_ep-0001.params\
  --voc12_root VOC2012\
  --la_crf_dir ${INITTRAINDIR}/${INITCAMCRFLA}\
  --ha_crf_dir ${INITTRAINDIR}/${INITCAMCRFHA}\
  --targetpixel_dir ${BOUNDARYPREDDIR}/boundary_fold${FOLD}_kernel${KERNEL}_threshold${THRESHOLD}\
  --gt_dir VOC2012/SegmentationClassAug\
  --base_dir voc12/trainaug_fold${FOLD}_base.npy\
  --novel_dir voc12/trainaug_fold${FOLD}_novel.npy\
  --session_name ${AFFWEIGHTDIR}/${AFFMETHOD}_${EXPID}_fold${FOLD}_affnet

## boundary-aware two-stage propatation (1st stage)
mkdir ${OUTRWDIR}/${INFERMETHOD}_${EXPID}_fold${FOLD}_affrw
mkdir ${OUTRWDIR}/${INFERMETHOD}_${EXPID}_fold${FOLD}_affrw_cam
python infer_aff_cam_boundary.py\
  --weights ${AFFWEIGHTDIR}/${AFFMETHOD}_${EXPID}_fold${FOLD}_affnet.pth\
  --infer_list voc12/train_aug.txt\
  --cam_dir ${INITCAMDIR}/${INITCAMTRAINAUG}\
  --voc12_root VOC2012\
  --out_rw ${OUTRWDIR}/${INFERMETHOD}_${EXPID}_fold${FOLD}_affrw\
  --out_rw_cam ${OUTRWDIR}/${INFERMETHOD}_${EXPID}_fold${FOLD}_affrw_cam\
  --targetpixel_dir ${BOUNDARYPREDDIR}/boundary_fold${FOLD}_kernel${KERNEL}_threshold${THRESHOLD}\
  --alpha 16\
  --beta 8\
  --logt 8

## boundary-aware two-stage propatation (2nd stage)
mkdir ${OUTRWDIR}/${INFERMETHOD}_${EXPID}_fold${FOLD}_affrw_secondary
mkdir ${OUTRWDIR}/${INFERMETHOD}_${EXPID}_fold${FOLD}_affrw_cam_secondary
python infer_aff_cam_boundary_secondary.py\
  --weights ${AFFWEIGHTDIR}/${AFFMETHOD}_${EXPID}_fold${FOLD}_affnet.pth\
  --infer_list voc12/train_aug.txt\
  --cam_dir ${OUTRWDIR}/${INFERMETHOD}_${EXPID}_fold${FOLD}_affrw_cam\
  --voc12_root VOC2012\
  --out_rw ${OUTRWDIR}/${INFERMETHOD}_${EXPID}_fold${FOLD}_affrw_secondary\
  --out_rw_cam ${OUTRWDIR}/${INFERMETHOD}_${EXPID}_fold${FOLD}_affrw_cam_secondary\
  --targetpixel_dir ${BOUNDARYPREDDIR}/boundary_fold${FOLD}_kernel${KERNEL}_threshold${THRESHOLD}\
  --alpha 16\
  --beta 8\
  --logt 8

## evaluate the propagation results
python evaluation_boundary.py\
  --list VOC2012/ImageSets/Segmentation/train.txt\
  --predict_dir ${OUTRWDIR}/${INFERMETHOD}_${EXPID}_fold${FOLD}_affrw_secondary\
  --targetpixel_dir None\
  --gt_dir VOC2012/SegmentationClassAug\
  --logfile ./evallog_${INFERMETHOD}/evallog_${EXPID}_fold${FOLD}.txt\
  --type png\
  --comment ${INFERMETHOD}_${EXPID}_fold${FOLD}_affrw2_train_full\
  --fold ${FOLD}

## evaluate the propagation results (search the background CAM with different thresholds)
python evaluation_boundary.py\
  --list VOC2012/ImageSets/Segmentation/train.txt\
  --predict_dir ${OUTRWDIR}/${INFERMETHOD}_${EXPID}_fold${FOLD}_affrw_cam_secondary\
  --targetpixel_dir None\
  --gt_dir VOC2012/SegmentationClassAug\
  --logfile ./evallog_${INFERMETHOD}/evallog_${EXPID}_fold${FOLD}.txt\
  --type npy_new\
  --curve True\
  --comment ${INFERMETHOD}_${EXPID}_fold${FOLD}_affrw2_train_full_searchbg\
  --fold ${FOLD}
