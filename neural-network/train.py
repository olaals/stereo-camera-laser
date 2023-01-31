import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from PIL import Image
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from unet import UNET
import shutil
from tqdm import tqdm


def apply_planar_homography(H, img):
    projected_img = cv2.warpPerspective(img, H, (img.shape[1], img.shape[0]))
    return projected_img

def read_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

class LaserDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path):
        calib_path = os.path.join(dataset_path, 'calib.json')
        calib_dict = read_json(calib_path)
        self.calib_dict = calib_dict
        self.H = calib_dict['H']
        img_dir = os.path.join(dataset_path, 'data')
        left_img_paths = [os.path.join(img_dir,'left', x) for x in os.listdir(os.path.join(img_dir, 'left'))]
        right_img_paths = [os.path.join(img_dir,'right', x) for x in os.listdir(os.path.join(img_dir, 'right'))]
        left_img_paths.sort()
        right_img_paths.sort()
        self.left_img_paths = left_img_paths
        self.right_img_paths = right_img_paths
        print("left_img_paths", len(left_img_paths))
        print("right_img_paths", len(right_img_paths))

    def __len__(self):
        return len(self.left_img_paths)

    def __getitem__(self, idx):
        left_img = cv2.imread(self.left_img_paths[idx], 0)
        right_img = cv2.imread(self.right_img_paths[idx], 0)
        print(left_img.shape)
        H = np.array(self.calib_dict['H'])
        proj_right_img = apply_planar_homography(H, right_img)
        left_img = left_img.astype(np.float32)/255.0
        proj_right_img = proj_right_img.astype(np.float32)/255.0
        left_img = torch.from_numpy(left_img)
        proj_right_img = torch.from_numpy(proj_right_img)
        left_img = left_img.unsqueeze(0)
        proj_right_img = proj_right_img.unsqueeze(0)
        return left_img, proj_right_img


def get_dataloaders(ds_path, train_val_split=0.95, batch_size=1):
    dataset = LaserDataset(ds_path)
    train_size = int(train_val_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def validate(model, val_loader, logdir, epoch):
    print("Validating")
    model.eval()
    epoch_val_dir = os.path.join(logdir, f'val_{epoch}')
    os.makedirs(epoch_val_dir, exist_ok=True)

    for idx,(left_img, right_img) in enumerate(val_loader):
        left_img= left_img.cuda()
        right_img = right_img.cuda()
        left_pred = model(left_img)
        right_pred = model(right_img)
        left_pred = torch.sigmoid(left_pred)
        right_pred = torch.sigmoid(right_pred)
        left_pred = tensor_to_img(left_pred)
        right_pred = tensor_to_img(right_pred)
        right_img = tensor_to_img(right_img)
        left_img = tensor_to_img(left_img)
        example_val_dir = os.path.join(epoch_val_dir, f'example_{idx}')
        os.makedirs(example_val_dir, exist_ok=True)
        cv2.imwrite(os.path.join(example_val_dir, 'left_img.png'), left_img)
        cv2.imwrite(os.path.join(example_val_dir, 'right_img.png'), right_img)
        cv2.imwrite(os.path.join(example_val_dir, 'left_pred.png'), left_pred)
        cv2.imwrite(os.path.join(example_val_dir, 'right_pred.png'), right_pred)

def tensor_to_img(prediction):
    prediction = prediction[0][0]
    prediction = prediction.cpu().detach().numpy()
    prediction = np.where(prediction > 255, 255, prediction)
    prediction = np.where(prediction <0, 0, prediction)
    prediction = (prediction * 255).astype(np.uint8)
    return prediction

def train():
    train_loader, val_loader = get_dataloaders('init-dataset-laser', batch_size=1)
    model = UNET(1, 1)
    model = model.cuda()
    logdir = 'logs'
    if os.path.exists(logdir):
        shutil.rmtree(logdir)
    os.makedirs(logdir)


    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(100):
        once_per_epoch = True
        print("Epoch", epoch)
        for left_img, right_img in tqdm(train_loader):
            left_img = left_img.cuda()
            right_img = right_img.cuda()
            optimizer.zero_grad()
            right_filt = model(right_img)
            right_loss = criterion(right_filt, left_img)
            right_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            left_filt = model(left_img)
            left_loss = criterion(left_filt, right_img)
            left_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if once_per_epoch:
                print("Epoch: ", epoch)
                print("left_loss: ", left_loss.item())
                print("right_loss: ", right_loss.item())
                once_per_epoch = False
                left_filt = torch.sigmoid(left_filt)
                right_filt = torch.sigmoid(right_filt)
                left_filt = tensor_to_img(left_filt)
                right_filt = tensor_to_img(right_filt)
                left_img = tensor_to_img(left_img)
                right_img = tensor_to_img(right_img)
                cv2.imwrite(os.path.join(logdir, f'left_img_{epoch}.png'), left_img)
                cv2.imwrite(os.path.join(logdir, f'right_img_{epoch}.png'), right_img)
                cv2.imwrite(os.path.join(logdir, f'left_filt_{epoch}.png'), left_filt)
                cv2.imwrite(os.path.join(logdir, f'right_filt_{epoch}.png'), right_filt)
                validate(model, val_loader, logdir, epoch)
                model.train()


            


                


def batch_to_imgs(batch, idx):
    tensor = batch[idx]
    print(tensor.shape)
    imgs = []
    for i in range(tensor.shape[0]):
        img = tensor[i].cpu().detach().numpy()
        # as uint8 image
        img = (img * 255).astype(np.uint8)
        imgs.append(img)
    return imgs




if __name__ == '__main__':
    train()

