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
from torch.utils.data import DataLoader
from models.swapnet import SwapNet, toTensor, var_to_np
from utils.util import get_image_paths, load_images, stack_images
from dataset.training_data import get_training_data
from alfred.dl.torch.common import device
from shutil import copyfile
from loguru import logger
from dataset.face_pair_dataset import FacePairDataset, FacePairDataset64x64
from torchvision import transforms
import sys

logger.remove()  # Remove the pre-configured handler
logger.start(sys.stderr, format="<lvl>{level}</lvl> {time:MM-DD HH:mm:ss} {file}:{line} - {message}")

batch_size = 64
epochs = 100000
save_per_epoch = 300

a_dir = './data/galgadot_fbb/fanbingbing_faces'
b_dir = './data/galgadot_fbb/galgadot_faces'
# we start to train on bigger size
target_size = 64
dataset_name = 'galgadot_fbb'
log_img_dir = './checkpoint/results_{}_{}x{}'.format(dataset_name, target_size, target_size)
log_model_dir = './checkpoint/{}_{}x{}'.format(dataset_name,
    target_size, target_size)
check_point_save_path = os.path.join(
    log_model_dir, 'faceswap_{}_{}x{}.pth'.format(dataset_name, target_size, target_size))


def main():
    os.makedirs(log_img_dir, exist_ok=True)
    os.makedirs(log_model_dir, exist_ok=True)

    transform = transforms.Compose([
        # transforms.Resize((target_size, target_size)),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.ToTensor(),
    ])
    ds = FacePairDataset64x64(a_dir=a_dir, b_dir=b_dir,
                         target_size=target_size, transform=transform)
    dataloader = DataLoader(ds, batch_size, shuffle=True)

    model = SwapNet()
    model.to(device)
    start_epoch = 0
    logger.info('try resume from checkpoint')
    if os.path.isdir('checkpoint'):
        try:
            if torch.cuda.is_available():
                checkpoint = torch.load(check_point_save_path)
            else:
                checkpoint = torch.load(
                    check_point_save_path, map_location={'cuda:0': 'cpu'})
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
    try:
        for epoch in range(start_epoch, epochs):
            iter = 0
            for data in dataloader:
                iter += 1
                img_a_target, img_a_input, img_b_target, img_b_input = data
                img_a_target = img_a_target.to(device)
                img_a_input = img_a_input.to(device)
                img_b_target = img_b_target.to(device)
                img_b_input = img_b_input.to(device)
                # print(img_a.size())
                # print(img_b.size())

                optimizer_1.zero_grad()
                optimizer_2.zero_grad()
                predict_a = model(img_a_input, select='A')
                predict_b = model(img_b_input, select='B')
                loss1 = criterion(predict_a, img_a_target)
                loss2 = criterion(predict_b, img_b_target)
                loss1.backward()
                loss2.backward()
                optimizer_1.step()
                optimizer_2.step()
                logger.info('Epoch: {}, iter: {}, lossA: {}, lossB: {}'.format(
                    epoch, iter, loss1.item(), loss2.item()))
                if epoch % save_per_epoch == 0 and epoch != 0:
                    logger.info('Saving models...')
                    state = {
                        'state': model.state_dict(),
                        'epoch': epoch
                    }
                    torch.save(state, os.path.join(os.path.dirname(
                        check_point_save_path), 'faceswap_trump_cage_128x128_{}.pth'.format(epoch)))
                    copyfile(os.path.join(os.path.dirname(check_point_save_path), 'faceswap_trump_cage_128x128_{}.pth'.format(epoch)),
                            check_point_save_path)
                if epoch % 10 == 0 and epoch != 0 and iter == 1:
                    img_a_original = np.array(img_a_target.detach().cpu().numpy()[0].transpose(2, 1, 0)*255, dtype=np.uint8)
                    img_b_original = np.array(img_b_target.detach().cpu().numpy()[0].transpose(2, 1, 0)*255, dtype=np.uint8)
                    a_predict_a = np.array(predict_a.detach().cpu().numpy()[0].transpose(2, 1, 0)*255, dtype=np.uint8)
                    b_predict_b = np.array(predict_b.detach().cpu().numpy()[0].transpose(2, 1, 0)*255, dtype=np.uint8)

                    a_predict_b = model(img_a_input, select='B')
                    b_predict_a = model(img_b_input, select='A')
                    a_predict_b = np.array(a_predict_b.detach().cpu().numpy()[0].transpose(2, 1, 0)*255, dtype=np.uint8)
                    b_predict_a = np.array(b_predict_a.detach().cpu().numpy()[0].transpose(2, 1, 0)*255, dtype=np.uint8)

                    cv2.imwrite(os.path.join(log_img_dir, '{}_0.png'.format(epoch)), cv2.cvtColor(img_a_original, cv2.COLOR_BGR2RGB))
                    cv2.imwrite(os.path.join(log_img_dir, '{}_3.png'.format(epoch)), cv2.cvtColor(img_b_original, cv2.COLOR_BGR2RGB))
                    cv2.imwrite(os.path.join(log_img_dir, '{}_1.png'.format(epoch)), cv2.cvtColor(a_predict_a, cv2.COLOR_BGR2RGB))
                    cv2.imwrite(os.path.join(log_img_dir, '{}_4.png'.format(epoch)), cv2.cvtColor(b_predict_b, cv2.COLOR_BGR2RGB))
                    cv2.imwrite(os.path.join(log_img_dir, '{}_2.png'.format(epoch)), cv2.cvtColor(a_predict_b, cv2.COLOR_BGR2RGB))
                    cv2.imwrite(os.path.join(log_img_dir, '{}_5.png'.format(epoch)), cv2.cvtColor(b_predict_a, cv2.COLOR_BGR2RGB))
                    logger.info('Record a result')
    except KeyboardInterrupt:
        logger.warning('try saving models...do not interrupt')
        state = {
            'state': model.state_dict(),
            'epoch': epoch
        }
        torch.save(state, os.path.join(os.path.dirname(
            check_point_save_path), 'faceswap_trump_cage_256x256_{}.pth'.format(epoch)))
        copyfile(os.path.join(os.path.dirname(check_point_save_path), 'faceswap_trump_cage_256x256_{}.pth'.format(epoch)),
                    check_point_save_path)



if __name__ == "__main__":
    main()
