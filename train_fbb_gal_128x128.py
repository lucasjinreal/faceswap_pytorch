"""
Copyright StrangeAI Authors @2019


As the network without linear connect layer
the feature are not compressed, so the encoder are weak
it consist to many informations, and decoder can not using the abstract 
information to construct a new image

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
from utils.util import get_image_paths, load_images, stack_images
from dataset.training_data import get_training_data
from alfred.dl.torch.common import device
from shutil import copyfile
try:
    from models.swapnet_128 import SwapNet128, toTensor, var_to_np
except Exception:
    print('can not import swapnet128, if you need high resolution face swap, '
    'you can download from http://luoli.ai (you can afford a VIP membership to get all other codes)')
from loguru import logger
from dataset.face_pair_dataset import FacePairDataset128x128
from torchvision import transforms
from torch.utils.data import DataLoader
from alfred.utils.log import init_logger

init_logger()

batch_size = 32
epochs = 100000
save_per_epoch = 300

a_dir = './data/galgadot_fbb/fanbingbing_faces'
b_dir = './data/galgadot_fbb/galgadot_faces'
# we start to train on bigger size
dataset_name = 'galgadot_fbb'
target_size = 128
log_img_dir = './checkpoint/results_{}_{}x{}'.format(dataset_name, target_size, target_size)
log_model_dir = './checkpoint/{}_{}x{}'.format(dataset_name,
    target_size, target_size)
check_point_save_path = os.path.join(
    log_model_dir, 'faceswap_{}_{}x{}.pth'.format(dataset_name, target_size, target_size))


def main():
    os.makedirs(log_img_dir, exist_ok=True)
    os.makedirs(log_model_dir, exist_ok=True)
    logger.info("loading datasets")

    transform = transforms.Compose([
        # transforms.Resize((target_size, target_size)),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.ToTensor(),
    ])
    ds = FacePairDataset128x128(a_dir=a_dir, b_dir=b_dir,
                         target_size=target_size, transform=transform)
    dataloader = DataLoader(ds, batch_size, shuffle=True)

    model = SwapNet128()
    model.to(device)
    start_epoch = 0
    logger.info('try resume from checkpoint')
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
        print('Can\'t found {}'.format(check_point_save_path))

    criterion = nn.L1Loss()
    optimizer_1 = optim.Adam([{'params': model.encoder.parameters()},
                              {'params': model.decoder_a.parameters()}], lr=5e-5, betas=(0.5, 0.999))
    optimizer_2 = optim.Adam([{'params': model.encoder.parameters()},
                              {'params': model.decoder_b.parameters()}], lr=5e-5, betas=(0.5, 0.999))

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
                predict_a = model(img_a_input, to='a')
                predict_b = model(img_b_input, to='b')
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
                        check_point_save_path), 'faceswap_{}_128x128_{}.pth'.format(dataset_name, epoch)))
                    copyfile(os.path.join(os.path.dirname(check_point_save_path), 'faceswap_{}_128x128_{}.pth'.format(dataset_name, epoch)),
                                    check_point_save_path)
                if epoch % 10 == 0 and epoch != 0 and iter == 1:
                    img_a_original = np.array(img_a_target.detach().cpu().numpy()[0].transpose(2, 1, 0)*255, dtype=np.uint8)
                    img_b_original = np.array(img_b_target.detach().cpu().numpy()[0].transpose(2, 1, 0)*255, dtype=np.uint8)
                    a_predict_a = np.array(predict_a.detach().cpu().numpy()[0].transpose(2, 1, 0)*255, dtype=np.uint8)
                    b_predict_b = np.array(predict_b.detach().cpu().numpy()[0].transpose(2, 1, 0)*255, dtype=np.uint8)

                    a_predict_b = model(img_a_input, to='b')
                    b_predict_a = model(img_b_input, to='a')
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
        logger.info('try saving models...')
        state = {
            'state': model.state_dict(),
            'epoch': epoch
        }
        torch.save(state, os.path.join(os.path.dirname(check_point_save_path), 'faceswap_{}_128x128_{}.pth'.format(dataset_name, epoch)))
        copyfile(os.path.join(os.path.dirname(check_point_save_path), 'faceswap_{}_128x128_{}.pth'.format(dataset_name, epoch)),
                        check_point_save_path)


if __name__ == "__main__":
    main()
