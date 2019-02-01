"""
grid a final image from result images

"""
import cv2
import numpy as np
import os
import sys
import glob
from PIL import Image


d = sys.argv[1]
print('from ', d)

all_img_files = glob.glob(os.path.join(d, '*.png'))
assert len(all_img_files) % 6 == 0, 'images divided by 6'
all_img_files = sorted(all_img_files)
rows = len(all_img_files) // 6
print(rows)
print(len(all_img_files))


res_img = Image.new('RGB',(128*6, 128*(len(all_img_files)//6)), (255, 255, 255))

for i in range(len(all_img_files)//6):
    for j in range(6):
        # print('now: ', all_img_files[6*i + j])
        img = Image.open(all_img_files[6*i + j])
        res_img.paste(img, (j*128, i*128))
res_img.save('res_grid.png')
print(np.array(res_img).shape)
cv2.imshow('rr', np.array(res_img))
cv2.waitKey(0)