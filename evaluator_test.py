import numpy as np
import os
from PIL import Image
from utils import Evaluator
import sys
eval_ = Evaluator(124)
eval_.reset()

DIR=sys.argv[1]
split = 'val.txt'

with open(os.path.join(DIR,split),'r') as f:
    lines = f.readlines()
    for line in lines:
        videolist = [line[:-1] for line in lines]
PRED=sys.argv[2]
for video in videolist:
    for tar in os.listdir(os.path.join(DIR,'data',video,'mask')):
        pred = os.path.join(PRED,video,tar)
        tar_ = Image.open(os.path.join(DIR,'data',video,'mask',tar))
        tar_ = np.array(tar_)
        tar_ = tar_[np.newaxis,:]
        pred_ = Image.open(pred)
        pred_ = np.array(pred_)
        pred_ = pred_[np.newaxis,:]
        eval_.add_batch(tar_,pred_)

Acc = eval_.Pixel_Accuracy()
Acc_class = eval_.Pixel_Accuracy_Class()
mIoU = eval_.Mean_Intersection_over_Union()
FWIoU = eval_.Frequency_Weighted_Intersection_over_Union()
print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))

