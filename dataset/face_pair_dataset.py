"""
Copyright StrangeAI authors @2019

assume you have to directly which you want
convert A to B, just put all faces of A person to A,
faces of B person to B

"""
import torch
from torch.utils.data import Dataset
import glob
import os
from alfred.dl.torch.common import device
import cv2
from PIL import Image
from torchvision import transforms
import numpy as np
from utils.umeyama import umeyama
import cv2

random_transform_args = {
    'rotation_range': 10,
    'zoom_range': 0.05,
    'shift_range': 0.05,
    'random_flip': 0.4,
}


def random_transform(image, rotation_range, zoom_range, shift_range, random_flip):
    h, w = image.shape[0:2]
    rotation = np.random.uniform(-rotation_range, rotation_range)
    scale = np.random.uniform(1 - zoom_range, 1 + zoom_range)
    tx = np.random.uniform(-shift_range, shift_range) * w
    ty = np.random.uniform(-shift_range, shift_range) * h
    mat = cv2.getRotationMatrix2D((w // 2, h // 2), rotation, scale)
    mat[:, 2] += (tx, ty)
    result = cv2.warpAffine(image, mat, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if np.random.random() < random_flip:
        result = result[:, ::-1]
    return result


def random_warp_128(image):
    assert image.shape == (256, 256, 3), 'resize image to 256 256 first'
    range_ = np.linspace(128 - 120, 128 + 120, 9)
    mapx = np.broadcast_to(range_, (9, 9))
    mapy = mapx.T
    mapx = mapx + np.random.normal(size=(9, 9), scale=5)
    mapy = mapy + np.random.normal(size=(9, 9), scale=5)
    interp_mapx = cv2.resize(mapx, (144, 144))[8:136, 8:136].astype('float32')
    interp_mapy = cv2.resize(mapy, (144, 144))[8:136, 8:136].astype('float32')
    warped_image = cv2.remap(image, interp_mapx, interp_mapy, cv2.INTER_LINEAR)
    src_points = np.stack([mapx.ravel(), mapy.ravel()], axis=-1)
    dst_points = np.mgrid[0:129:16, 0:129:16].T.reshape(-1, 2)
    mat = umeyama(src_points, dst_points, True)[0:2]
    target_image = cv2.warpAffine(image, mat, (128, 128))
    return warped_image, target_image


def random_warp_64(image):
    assert image.shape == (256, 256, 3)
    range_ = np.linspace(128 - 120, 128 + 120, 5)
    mapx = np.broadcast_to(range_, (5, 5))
    mapy = mapx.T
    mapx = mapx + np.random.normal(size=(5, 5), scale=5)
    mapy = mapy + np.random.normal(size=(5, 5), scale=5)
    interp_mapx = cv2.resize(mapx, (80, 80))[8:72, 8:72].astype('float32')
    interp_mapy = cv2.resize(mapy, (80, 80))[8:72, 8:72].astype('float32')
    warped_image = cv2.remap(image, interp_mapx, interp_mapy, cv2.INTER_LINEAR)
    src_points = np.stack([mapx.ravel(), mapy.ravel()], axis=-1)
    dst_points = np.mgrid[0:65:16, 0:65:16].T.reshape(-1, 2)
    mat = umeyama(src_points, dst_points, True)[0:2]
    target_image = cv2.warpAffine(image, mat, (64, 64))
    return warped_image, target_image


class FacePairDataset(Dataset):

    def __init__(self, a_dir, b_dir, target_size, transform):
        super(FacePairDataset, self).__init__
        self.a_dir = a_dir
        self.b_dir = b_dir
        self.target_size = target_size

        self.transform = transform
        # extension can be changed here to png or others
        self.a_images_list = glob.glob(os.path.join(a_dir, '*.png'))
        self.b_images_list = glob.glob(os.path.join(b_dir, '*.png'))

    def __getitem__(self, index):
        # return 2 image pair, A and B
        img_a = Image.open(self.a_images_list[index])
        img_b = Image.open(self.b_images_list[index])
        
        # align the face first
        img_a = img_a.resize((self.target_size, self.target_size), Image.ANTIALIAS)
        img_b = img_b.resize((self.target_size, self.target_size), Image.ANTIALIAS)
       
        # transform
        if self.transform:
            img_a = self.transform(img_a)
            img_b = self.transform(img_b)
  
        # already resized, warp it
        img_a = random_transform(np.array(img_a), **random_transform_args)
        img_b = random_transform(np.array(img_b), **random_transform_args)
        img_a_input, img_a = random_warp(np.array(img_a), 256)
        img_b_input, img_b = random_warp(np.array(img_b), 256)
        img_a_tensor = torch.Tensor(img_a.transpose(2, 0, 1)/255.).float()
        img_a_input_tensor = torch.Tensor(img_a_input.transpose(2, 0, 1)/255.).float()
        img_b_tensor = torch.Tensor(img_b.transpose(2, 0, 1)/255.).float()
        img_b_input_tensor = torch.Tensor(img_b_input.transpose(2, 0, 1)/255.).float()
        return img_a_tensor, img_a_input_tensor, img_b_tensor, img_b_input_tensor

    def __len__(self):
        return min(len(self.a_images_list), len(self.b_images_list))



class FacePairDataset64x64(Dataset):

    def __init__(self, a_dir, b_dir, target_size, transform):
        super(FacePairDataset64x64, self).__init__
        self.a_dir = a_dir
        self.b_dir = b_dir
        self.target_size = target_size

        self.transform = transform
        # extension can be changed here to png or others
        self.a_images_list = glob.glob(os.path.join(a_dir, '*.png'))
        self.b_images_list = glob.glob(os.path.join(b_dir, '*.png'))

    def __getitem__(self, index):
        # return 2 image pair, A and B
        img_a = Image.open(self.a_images_list[index])
        img_b = Image.open(self.b_images_list[index])
        
        # align the face first
        img_a = img_a.resize((256, 256), Image.ANTIALIAS)
        img_b = img_b.resize((256, 256), Image.ANTIALIAS)
       
        # transform
        if self.transform:
            img_a = self.transform(img_a)
            img_b = self.transform(img_b)
  
        # # already resized, warp it
        img_a = random_transform(np.array(img_a), **random_transform_args)
        img_b = random_transform(np.array(img_b), **random_transform_args)
        img_a_input, img_a = random_warp_64(np.array(img_a))
        img_b_input, img_b = random_warp_64(np.array(img_b))

        img_a = np.array(img_a)
        img_b = np.array(img_b)

        img_a_tensor = torch.Tensor(img_a.transpose(2, 0, 1)/255.).float()
        img_a_input_tensor = torch.Tensor(img_a_input.transpose(2, 0, 1)/255.).float()
        img_b_tensor = torch.Tensor(img_b.transpose(2, 0, 1)/255.).float()
        img_b_input_tensor = torch.Tensor(img_b_input.transpose(2, 0, 1)/255.).float()
        return img_a_tensor, img_a_input_tensor, img_b_tensor, img_b_input_tensor

    def __len__(self):
        return min(len(self.a_images_list), len(self.b_images_list))



class FacePairDataset128x128(Dataset):

    def __init__(self, a_dir, b_dir, target_size, transform):
        super(FacePairDataset128x128, self).__init__
        self.a_dir = a_dir
        self.b_dir = b_dir
        self.target_size = target_size

        self.transform = transform
        self.a_images_list = glob.glob(os.path.join(a_dir, '*.png'))
        self.b_images_list = glob.glob(os.path.join(b_dir, '*.png'))

    def __getitem__(self, index):
        # return 2 image pair, A and B
        img_a = Image.open(self.a_images_list[index])
        img_b = Image.open(self.b_images_list[index])
        
        # align the face first
        img_a = img_a.resize((256, 256), Image.ANTIALIAS)
        img_b = img_b.resize((256, 256), Image.ANTIALIAS)
       
        # transform
        if self.transform:
            img_a = self.transform(img_a)
            img_b = self.transform(img_b)
  
        img_a = random_transform(np.array(img_a), **random_transform_args)
        img_b = random_transform(np.array(img_b), **random_transform_args)
        img_a_input, img_a = random_warp_128(np.array(img_a))
        img_b_input, img_b = random_warp_128(np.array(img_b))
        img_a_tensor = torch.Tensor(img_a.transpose(2, 0, 1)/255.).float()
        img_a_input_tensor = torch.Tensor(img_a_input.transpose(2, 0, 1)/255.).float()
        img_b_tensor = torch.Tensor(img_b.transpose(2, 0, 1)/255.).float()
        img_b_input_tensor = torch.Tensor(img_b_input.transpose(2, 0, 1)/255.).float()
        return img_a_tensor, img_a_input_tensor, img_b_tensor, img_b_input_tensor

    def __len__(self):
        return min(len(self.a_images_list), len(self.b_images_list))