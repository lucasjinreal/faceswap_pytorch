"""
convert  a face to another person

"""
from models.swapnet import SwapNet
import torch
from alfred.dl.torch.common import device
import cv2
import numpy as np
from dataset.training_data import random_warp
from utils.umeyama import umeyama

mean_value = np.array([0.03321508, 0.05035182, 0.02038819])


def process_img(ori_img):
    img = cv2.resize(ori_img, (256, 256))
    range_ = np.linspace( 128-80, 128+80, 5 )
    mapx = np.broadcast_to( range_, (5,5) )
    mapy = mapx.T

    # warp image like in the training
    mapx = mapx + np.random.normal( size=(5,5), scale=5 )
    mapy = mapy + np.random.normal( size=(5,5), scale=5 )
    interp_mapx = cv2.resize(mapx, (80, 80))[8:72, 8:72].astype('float32')
    interp_mapy = cv2.resize(mapy, (80, 80))[8:72, 8:72].astype('float32')
    warped_image = cv2.remap(img, interp_mapx, interp_mapy, cv2.INTER_LINEAR)
    return warped_image


def load_img():
    a = 'images/34600_test_A_target.png'
    img = cv2.imread(a) / 255.
    return img


def predict():
    # convert trump to cage
    # img_f = 'data/trump/51834796.jpg'
    # img_f = 'data/trump/494045244.jpg'
    # NOTE: using face extracted (not original image)
    img_f = 'data/trump/464669134_face_0.png'

    ori_img = cv2.imread(img_f)
    img = cv2.resize(ori_img, (64, 64)) / 255.
    img = np.rot90(img)
    # img = load_img()
    in_img = np.array(img, dtype=np.float).transpose(2, 1, 0)

    # normalize img
    in_img = torch.Tensor(in_img).to(device).unsqueeze(0)
    model = SwapNet().to(device)
    if torch.cuda.is_available():
        checkpoint = torch.load('checkpoint/faceswap_trump_cage_64x64.pth')
    else:
        checkpoint = torch.load('checkpoint/faceswap_trump_cage_64x64.pth', map_location={'cuda:0': 'cpu'})
    model.load_state_dict(checkpoint['state'])
    model.eval()
    print('model loaded.')

    out = model.forward(in_img, select='B')
    out = np.clip(out.detach().cpu().numpy()[0]*255, 0, 255).astype('uint8').transpose(2, 1, 0)

    cv2.imshow('original image', ori_img)
    cv2.imshow('network input image', img)
    cv2.imshow('result image', np.rot90(out, axes=(1, 0)))
    cv2.waitKey(0)


if __name__ == '__main__':
    predict()
