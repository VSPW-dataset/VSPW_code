import numpy as np
import os
from PIL import Image
#from utils import Evaluator


def get_common(list_,predlist,clip_num,h,w):
    accs = []
    for i in range(len(list_)-clip_num):
        global_common = np.ones((h,w))
        predglobal_common = np.ones((h,w))

                 
        for j in range(1,clip_num):
            common = (list_[i] == list_[i+j])
            global_common = np.logical_and(global_common,common)
            pred_common = (predlist[i]==predlist[i+j])
            predglobal_common = np.logical_and(predglobal_common,pred_common)
        pred = (predglobal_common*global_common)

        acc = pred.sum()/global_common.sum()
        accs.append(acc)
    return accs
        


DIR='/home/miaojiaxu/jiaxu_2/semantic_seg/LVSP_plus_data_label124_480p'
#tar_dir = '/home/miaojiaxu/jiaxu_2/semantic_seg/newsaveimages'
tar_dir = '/home/miaojiaxu/jiaxu_2/semantic_seg/newsaveimages_ab/abpsp'
#for clip_num in [8,16]:
for clip_num in [16]:
#    for split in ['val.txt','test.txt']:
    for split in ['val.txt']:
        with open(os.path.join(DIR,split),'r') as f:
            lines = f.readlines()
            for line in lines:
                videolist = [line[:-1] for line in lines]
        for fold in os.listdir(tar_dir):
            if fold =='__pycache__':
                continue
            if os.path.isdir(os.path.join(tar_dir,fold)):
                Pred = os.path.join(tar_dir,fold)
            else:
                continue
        #Pred='/home/miaojiaxu/jiaxu_2/semantic_seg/VSP_124_saveimg/clip_ocr_369_result1'
        
        
        
        
            total_acc=[]
            for video in videolist:
                imglist = []
                predlist = []
            
                images = sorted(os.listdir(os.path.join(DIR,'data',video,'mask')))
            
                if len(images)<=clip_num:
                    continue
                for imgname in images:
                    img = Image.open(os.path.join(DIR,'data',video,'mask',imgname))
                    w,h = img.size
                    img = np.array(img)
                    imglist.append(img)
                    pred = Image.open(os.path.join(Pred,video,imgname))
                    pred = np.array(pred)
                    predlist.append(pred)
                     
                accs = get_common(imglist,predlist,clip_num,h,w)
#                print(sum(accs)/len(accs))
                total_acc.extend(accs)
            Acc = np.array(total_acc)
            Acc = np.nanmean(Acc)
            print('*'*10)
            print(Acc)
            print(clip_num)
            print(Pred)
            print(split)
    
