import os
import pandas as pd
import numpy as np
from PIL import Image
import multiprocessing
import argparse

categories = ['background','aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow',
              'diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
novel_categories = ['pottedplant','sheep','sofa','train','tvmonitor']

def do_python_eval(predict_folder, targetpixel_folder, gt_folder, output_folder, name_list, num_cls=21, input_type='png', threshold=1.0, printlog=False):
    TP = []
    P = []
    T = []
    for i in range(num_cls):
        TP.append(multiprocessing.Value('i', 0, lock=True))
        P.append(multiprocessing.Value('i', 0, lock=True))
        T.append(multiprocessing.Value('i', 0, lock=True))
    
    def compare(start,step,TP,P,T,input_type,threshold):
        for idx in range(start,len(name_list),step):
            name = name_list[idx]
            if input_type == 'npy':
                predict_file = os.path.join(predict_folder,'%s.npy'%name)
                predict_dict = np.load(predict_file, allow_pickle=True).item()
                h, w = list(predict_dict.values())[0].shape
                tensor = np.zeros((21,h,w),np.float32)
                for key in predict_dict.keys():
                    tensor[key+1] = predict_dict[key]
                tensor[0,:,:] = threshold 
                predict = np.argmax(tensor, axis=0).astype(np.uint8)
            elif input_type == 'npy_new':
                predict_file = os.path.join(predict_folder,'%s.npy'%name)
                predict_dict = np.load(predict_file, allow_pickle=True).item()
                h, w = list(predict_dict.values())[0].shape
                tensor = np.zeros((21,h,w),np.float32)
                for key in predict_dict.keys():
                    if key == 0:
                        continue
                    tensor[key] = predict_dict[key]
                tensor[0,:,:] = threshold 
                predict = np.argmax(tensor, axis=0).astype(np.uint8)

            Image.fromarray(predict).save(os.path.join(output_folder,'%s.png'%name))

            gt_file = os.path.join(gt_folder,'%s.png'%name)
            gt = np.array(Image.open(gt_file)).astype(np.uint8)
            ## start add ##
            if targetpixel_folder != 'None':
                targetpixel_file = os.path.join(targetpixel_folder,'%s.png'%name)
                targetpixel = np.array(Image.open(targetpixel_file)).astype(np.uint8)
                gt = np.where(targetpixel == 0, 255, gt).astype(np.uint8)
            ##  end  add ##
            cal = gt<255
            mask = (predict==gt) * cal
      
            for i in range(num_cls):
                P[i].acquire()
                P[i].value += np.sum((predict==i)*cal)
                P[i].release()
                T[i].acquire()
                T[i].value += np.sum((gt==i)*cal)
                T[i].release()
                TP[i].acquire()
                TP[i].value += np.sum((gt==i)*mask)
                TP[i].release()
    p_list = []
    for i in range(8):
        p = multiprocessing.Process(target=compare, args=(i,8,TP,P,T,input_type,threshold))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()
    IoU = []
    T_TP = []
    P_TP = []
    FP_ALL = []
    FN_ALL = [] 
    for i in range(num_cls):
        IoU.append(TP[i].value/(T[i].value+P[i].value-TP[i].value+1e-10))
        T_TP.append(T[i].value/(TP[i].value+1e-10))
        P_TP.append(P[i].value/(TP[i].value+1e-10))
        FP_ALL.append((P[i].value-TP[i].value)/(T[i].value + P[i].value - TP[i].value + 1e-10))
        FN_ALL.append((T[i].value-TP[i].value)/(T[i].value + P[i].value - TP[i].value + 1e-10))
    loglist = {}
    for i in range(num_cls):
        loglist[categories[i]] = IoU[i] * 100
               
    miou = np.mean(np.array(IoU))
    loglist['mIoU'] = miou * 100

    novel_mIoU = np.mean(np.array([loglist[cat] for cat in novel_categories]))
    loglist['novel-mIoU'] = novel_mIoU

    if printlog:
        for i in range(num_cls):
            if i%2 != 1:
                print('%11s:%7.3f%%'%(categories[i],IoU[i]*100),end='\t')
            else:
                print('%11s:%7.3f%%'%(categories[i],IoU[i]*100))
        print('\n======================================================')
        print('%11s:%7.3f%%'%('mIoU',miou*100))
        print('%11s:%7.3f%%'%('novel-mIoU',novel_mIoU))
    return loglist


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--list", default='./VOC2012/ImageSets/SegmentationAug/train_aug_new_-5.txt', type=str)
    parser.add_argument("--predict_dir", default='./result1/boundary1.0_5_0.02_0.085_2.0_1_label', type=str)
    parser.add_argument("--targetpixel_dir", default='None', type=str)
    parser.add_argument("--gt_dir", default='./VOC2012/SegmentationClassAug', type=str)
    parser.add_argument("--output_dir", default='None', type=str)
    parser.add_argument('--type', default='png', choices=['npy', 'npy_new'], type=str)
    parser.add_argument('--t', default=0.26, type=float)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    df = pd.read_csv(args.list, names=['filename'])
    name_list = df['filename'].values
    loglist = do_python_eval(args.predict_dir, args.targetpixel_dir, args.gt_dir, args.output_dir, name_list, 21, args.type, args.t)
    print('background score: %.3f\tmIoU: %.3f%% novel-mIoU: %.3f%%'%(args.t, loglist['mIoU'], loglist['novel-mIoU']))
