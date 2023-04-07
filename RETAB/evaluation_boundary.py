import os
import pandas as pd
import numpy as np
from PIL import Image
import multiprocessing
import argparse

categories = ['background','aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow',
              'diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']

def do_python_eval(predict_folder, targetpixel_folder, gt_folder, name_list, base_categories, novel_categories, num_cls=21, input_type='png', threshold=1.0, printlog=False):

    assert (len(base_categories) + len(novel_categories) == num_cls)
    for i in categories:
        assert (i in base_categories or i in novel_categories)

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
            if input_type == 'png':
                predict_file = os.path.join(predict_folder,'%s.png'%name)
                predict = np.array(Image.open(predict_file)) #cv2.imread(predict_file)
            elif input_type == 'npy':
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

    base_mIoU = np.mean(np.array([loglist[cat] for cat in base_categories]))
    loglist['base-mIoU'] = base_mIoU

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
        print('%11s:%7.3f%%'%('base-mIoU',base_mIoU))
        print('%11s:%7.3f%%'%('novel-mIoU',novel_mIoU))
    return loglist

def writedict(file, dictionary):
    s = ''
    for key in dictionary.keys():
        sub = '%s:%s  '%(key, dictionary[key])
        s += sub
    s += '\n'
    file.write(s)

def writelog(filepath, metric, comment):
    filepath = filepath
    logfile = open(filepath,'a')
    import time
    logfile.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    logfile.write('\t%s\n'%comment)
    writedict(logfile, metric)
    if 'background_score' in metric.keys():
        logfile.write('{0:.2f}/{1:.2f}/{2:.2f}, bgscore:{3}\n'.format(metric['mIoU'], metric['base-mIoU'], metric['novel-mIoU'], metric['background_score']))
    else:
        logfile.write('{0:.2f}/{1:.2f}/{2:.2f}\n'.format(metric['mIoU'], metric['base-mIoU'], metric['novel-mIoU']))
    logfile.write('=====================================\n')
    logfile.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--list", default='./VOC2012/ImageSets/Segmentation/train.txt', type=str)
    parser.add_argument("--predict_dir", default='./pred', type=str)
    parser.add_argument("--targetpixel_dir", default='None', type=str)
    parser.add_argument("--gt_dir", default='./VOC2012/SegmentationClassAug', type=str)
    parser.add_argument('--logfile', default='./evallog.txt',type=str)
    parser.add_argument('--comment', required=True, type=str)
    parser.add_argument('--type', default='png', choices=['npy', 'png', 'npy_new'], type=str)
    parser.add_argument('--t', default=None, type=float)
    parser.add_argument('--curve', default=False, type=bool)
    parser.add_argument('--fold', required=True, choices=[0, 1, 2, 3], type=int)
    args = parser.parse_args()

    novel_cls = list(range(args.fold*5+1, (args.fold+1)*5+1))
    base_cls = [i for i in range(21) if i not in novel_cls]
    novel_categories = [categories[i] for i in novel_cls]
    base_categories = [categories[i] for i in base_cls]

    if args.type == 'npy' or args.type == 'npy_new':
        assert args.t is not None or args.curve
    df = pd.read_csv(args.list, names=['filename'])
    name_list = df['filename'].values
    if not args.curve:
        loglist = do_python_eval(args.predict_dir, args.targetpixel_dir, args.gt_dir, name_list, base_categories, novel_categories, 21, args.type, args.t, printlog=True)
        writelog(args.logfile, loglist, args.comment)
    else:
        resall = []
        novelall = []
        loglistall = []
        start_id, end_id = 0, 41
        for i in range(start_id, end_id+1):
            t = i/100.0
            loglist = do_python_eval(args.predict_dir, args.targetpixel_dir, args.gt_dir, name_list, base_categories, novel_categories, 21, args.type, t)
            resall.append(loglist['mIoU'])
            novelall.append(loglist['novel-mIoU'])
            print('%d/60 background score: %.3f\tmIoU: %.3f%% base-mIoU: %.3f%% novel-mIoU: %.3f%%'%(i, t, loglist['mIoU'], loglist['base-mIoU'], loglist['novel-mIoU']))
            loglist['background_score'] = '%.3f'%(t)
            loglistall.append(loglist)
        index = np.argmax(np.array(resall))
        if index == 0 or index == end_id-start_id:
            writelog(args.logfile, {'wrong': 'wrong'}, args.comment)
        else:
            writelog(args.logfile, loglistall[index], args.comment)
        index = np.argmax(np.array(novelall))
        if index == 0 or index == end_id-start_id:
            writelog(args.logfile, {'wrong': 'wrong'}, args.comment)
        else:
            writelog(args.logfile, loglistall[index], args.comment)
