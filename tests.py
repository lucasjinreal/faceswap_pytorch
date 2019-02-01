from models.swapnet import SwapNet
from models.swapnet_128 import SwapNet128
from utils.model_summary import summary
from alfred.dl.torch.common import device
from dataset.face_pair_dataset import random_warp_128
from dataset.training_data import  random_transform, random_transform_args
from PIL import Image
import cv2
import numpy as np
import torch
from utils.umeyama import umeyama

# model = SwapNet().to(device)
# summary(model, input_size=(3, 64, 64))

# def random_warp(image):
#     assert image.shape == (256, 256, 3)
#     range_ = np.linspace(128 - 120, 128 + 120, 5)
#     mapx = np.broadcast_to(range_, (5, 5))
#     mapy = mapx.T
#     mapx = mapx + np.random.normal(size=(5, 5), scale=5)
#     mapy = mapy + np.random.normal(size=(5, 5), scale=5)

#     interp_mapx = cv2.resize(mapx, (80, 80))[8:72, 8:72].astype('float32')
#     interp_mapy = cv2.resize(mapy, (80, 80))[8:72, 8:72].astype('float32')

#     # just crop the image, remove the top left bottom right 8 pixels (in order to get the pure face)
#     warped_image = cv2.remap(image, interp_mapx, interp_mapy, cv2.INTER_LINEAR)

#     src_points = np.stack([mapx.ravel(), mapy.ravel()], axis=-1)
#     dst_points = np.mgrid[0:65:16, 0:65:16].T.reshape(-1, 2)
#     mat = umeyama(src_points, dst_points, True)[0:2]
#     target_image = cv2.warpAffine(image, mat, (64, 64))
#     return warped_image, target_image

# model = SwapNet128().to(device)
# summary(model, input_size=(3, 128, 128))

# a = Image.open('data/trump_cage/cage/2455911_face_0.png')
# a = a.resize((256, 256), Image.ANTIALIAS)
# a = random_transform(np.array(a), **random_transform_args)
# warped_img, target_img = random_warp_128(np.array(a))

# t = torch.from_numpy(target_img.transpose(2, 0, 1) / 255.).to(device)
# b = t.detach().cpu().numpy().transpose((2, 1, 0))*255
# print(b.shape)

# cv2.imshow('rr', np.array(a))
# cv2.imshow('warped image', np.array(warped_img))
# cv2.imshow('target image', np.array(target_img))
# cv2.imshow('bbbbbbbbb', b)
# cv2.waitKey(0)