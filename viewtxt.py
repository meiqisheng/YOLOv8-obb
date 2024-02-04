import os
import os.path as osp
import cv2
import numpy as np
import shutil

BASEDIR = osp.abspath('.')
IMGDIR = osp.join(BASEDIR, 'images')
LABELDIR = osp.join(BASEDIR, 'labelTxt')
OUTDIR = osp.join(BASEDIR, 'watch_GT')
if osp.exists(OUTDIR):
    shutil.rmtree(OUTDIR)
os.makedirs(OUTDIR)


key_list = [name.split('.')[-2] for name in os.listdir(IMGDIR) if name.endswith('.png')]

print(key_list)
for k in key_list:
    imgname = k+'.png' 
    labelname = k+'.txt'
    path_img = osp.join(IMGDIR, imgname)
    path_label = osp.join(LABELDIR, labelname)
    img = cv2.imread(path_img)
    with open(path_label, 'r') as fp:
       lines = fp.readlines()
    # print(lines)
    for line in lines:
        one_box_info = line.strip().split()
        *xy4, cls_name, easy_token = one_box_info
        xy4 = list(map(int, xy4))
        xy4 = [[xy4[0],xy4[1]],
               [xy4[2],xy4[3]],
               [xy4[4],xy4[5]],
               [xy4[6],xy4[7]]]
        xy4 = np.array(xy4)
        # print(xy4)
        cv2.drawContours(img, [xy4], 0, color=(255, 255, 0), thickness=3)
        
    path_out = osp.join(OUTDIR, imgname.split('.')[-2] + '_watchGT.' + imgname.split('.')[-1])
    cv2.imwrite(path_out, img)
    print(imgname, 'done.')