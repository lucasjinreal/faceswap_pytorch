"""
Copyright StrangeAI Authors @2019


original forked from deepfakes repo

edit and promoted by StrangeAI authors

"""

from __future__ import print_function
import argparse
import os

import cv2
import numpy as np
import torch

import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
import torch.backends.cudnn as cudnn

from models.swapnet import SwapNet, toTensor, var_to_np
from utils.util import get_image_paths, load_images, stack_images
from dataset.training_data import get_training_data
from alfred.dl.torch.common import device
from shutil import copyfile
from loguru import logger

batch_size = 64
epochs = 100000
save_per_epoch = 300

a_dir = './data/trump_cage/trump'
b_dir = './data/trump_cage/cage'
# we start to train on bigger size
target_size = 64
dataset_name = 'trump_cage'
log_img_dir = './checkpoint/results_{}_{}x{}'.format(dataset_name, target_size, target_size)
log_model_dir = './checkpoint/{}_{}x{}'.format(dataset_name,
    target_size, target_size)
check_point_save_path = os.path.join(
    log_model_dir, 'faceswap_{}_{}x{}.pth'.format(dataset_name, target_size, target_size))


def main():
    os.makedirs(log_img_dir, exist_ok=True)
    os.makedirs(log_model_dir, exist_ok=True)
    
    logger.info("loading datasets")
    images_A = get_image_paths(a_dir)
    images_B = get_image_paths(b_dir)
    images_A = load_images(images_A) / 255.0
    images_B = load_images(images_B) / 255.0

    print('mean value to remember: ', images_B.mean(
        axis=(0, 1, 2)) - images_A.mean(axis=(0, 1, 2)))
    images_A += images_B.mean(axis=(0, 1, 2)) - images_A.mean(axis=(0, 1, 2))

    model = SwapNet()
    model.to(device)
    start_epoch = 0
    logger.info('try resume from checkpoint')
    if os.path.isdir('checkpoint'):
        try:
            if torch.cuda.is_available():
                checkpoint = torch.load('./checkpoint/faceswap_trump_cage_64x64.pth')
            else:
                checkpoint = torch.load(
                    './checkpoint/faceswap_trump_cage_64x64.pth', map_location={'cuda:0': 'cpu'})
            model.load_state_dict(checkpoint['state'])
            start_epoch = checkpoint['epoch']
            logger.info('checkpoint loaded.')
        except FileNotFoundError:
            print('Can\'t found faceswap_trump_cage.pth')

    criterion = nn.L1Loss()
    optimizer_1 = optim.Adam([{'params': model.encoder.parameters()},
                              {'params': model.decoder_A.parameters()}], lr=5e-5, betas=(0.5, 0.999))
    optimizer_2 = optim.Adam([{'params': model.encoder.parameters()},
                              {'params': model.decoder_B.parameters()}], lr=5e-5, betas=(0.5, 0.999))

    logger.info('Start training, from epoch {} '.format(start_epoch))

    for epoch in range(start_epoch, epochs):
        warped_A, target_A = get_training_data(images_A, batch_size)
        # print(warped_A.shape)
        # t_a = np.array(warped_A[0] * 255, dtype=np.uint8)
        # print(t_a)
        # print(t_a.shape)
        # cv2.imshow('rr', t_a)
        # cv2.waitKey(0)
        # warped a and target a are not rotated, where did rotate?

        warped_B, target_B = get_training_data(images_B, batch_size)
        warped_A, target_A = toTensor(warped_A), toTensor(target_A)
        warped_B, target_B = toTensor(warped_B), toTensor(target_B)
        # warp_a = np.array(warped_A[0].detach().cpu().numpy().transpose(2, 1, 0)*255, dtype=np.uint8)
        # cv2.imshow('rr', warp_a)
        # cv2.waitKey(0)
        warped_A, target_A, warped_B, target_B = Variable(warped_A.float()), Variable(target_A.float()), \
            Variable(warped_B.float()), Variable(target_B.float())
        optimizer_1.zero_grad()
        optimizer_2.zero_grad()
        warped_A_out = model(warped_A, 'A')
        warped_B_out = model(warped_B, 'B')
        loss1 = criterion(warped_A_out, target_A)
        loss2 = criterion(warped_B_out, target_B)
        loss1.backward()
        loss2.backward()
        optimizer_1.step()
        optimizer_2.step()
        logger.info('epoch: {}, lossA: {}, lossB: {}'.format(epoch, loss1.item(), loss2.item()))
        if epoch % save_per_epoch == 0 and iter == 0:
            logger.info('Saving models...')
            state = {
                'state': model.state_dict(),
                'epoch': epoch
            }
            torch.save(state, os.path.join(os.path.dirname(
                        check_point_save_path), 'faceswap_{}_64x64_{}.pth'.format(dataset_name, epoch)))
            copyfile(os.path.join(os.path.dirname(check_point_save_path), 'faceswap_{}_64x64_{}.pth'.format(dataset_name, epoch)),
                            check_point_save_path)
        if epoch % 100 == 0:
            test_A_ = warped_A[0:2]   
            a_predict_a = var_to_np(model(test_A_, 'A'))[0]*255
            # warped a out
            # print(test_A_[0].detach().cpu().numpy().shape)
            a_predict_b = var_to_np(model(test_A_, 'B'))[0]*255

            warp_a = test_A_[0].detach().cpu().numpy()*255
            target_a = target_A[0].detach().cpu().numpy()*255

            cv2.imwrite(os.path.join(log_img_dir, "{}_res_a_to_a.png".format(epoch)), np.array(a_predict_a.transpose(2, 1, 0)).astype('uint8'))
            cv2.imwrite(os.path.join(log_img_dir, "{}_res_a_to_b.png".format(epoch)), np.array(a_predict_b.transpose(2, 1, 0)).astype('uint8'))
            cv2.imwrite(os.path.join(log_img_dir, "{}_test_A_warped.png".format(epoch)), np.array(warp_a.transpose(2, 1, 0)).astype('uint8'))
            cv2.imwrite(os.path.join(log_img_dir, "{}_test_A_target.png".format(epoch)), np.array(target_a.transpose(2, 1, 0)).astype('uint8'))
            logger.info('Record a result')


if __name__ == "__main__":
    main()
