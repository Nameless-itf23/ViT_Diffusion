
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import random

def dataloader(batch_size:int=64, data_num:int=10000, denoise_num:int=500, sigma:float=0.3):

    data_path = './data'
    image_paths = [os.path.join(data_path, f) for f in os.listdir(data_path)]

    images = []
    for i, path in enumerate(image_paths):
        img = Image.open(path)
        img_array = np.array(img)
        images.append(img_array)
        print(f'\r{i+1}/{len(image_paths)}',end='')
    print('')

    images = np.stack(images)
    images = np.transpose(images, (0, 3, 1, 2))
    images = images / 255

    def rebin(a, shape):
        sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
        return a.reshape(sh).mean(-1).mean(1)

    tmp_images = []
    for i, image in enumerate(images):
        dims = []
        for dim in image:
            dims.append(rebin(dim, (32, 32)))
        tmp_images.append(np.stack(dims))
        print(f'\r{i+1}/{images.shape[0]}',end='')
    print('')
    images = np.array(tmp_images)

    T = denoise_num

    def noise_scheduler(t):  # t -> sigma
        return 1 - t / T

    xs = []
    ts = []
    ys = []

    for num in range(data_num):
        tmp_image = images[random.randrange(len(images))].copy()
        tmp_noise = np.random.normal(0.5, sigma, np.shape(tmp_image))
        t = random.randrange(T - 1)
        tmp_xs = noise_scheduler(t+1) * tmp_image + (1 - noise_scheduler(t+1)) * tmp_noise
        tmp_ys = noise_scheduler(t) * tmp_image + (1 - noise_scheduler(t)) * tmp_noise
        xs.append(tmp_xs)
        ts.append([t])
        ys.append(tmp_ys)
        print(f'\r{num+1}/{data_num}',end='')
    print('')

    xs = np.array(xs)
    ts = np.array(ts)
    ys = np.array(ys)


    class MyDataset(Dataset):
        def __init__(self, xs, ts, ys):
            self.xs = xs
            self.ts = ts
            self.ys = ys

        def __len__(self):
            return len(self.xs)

        def __getitem__(self, idx):
            x = torch.FloatTensor(self.xs[idx])
            t = torch.LongTensor(self.ts[idx])
            y = torch.FloatTensor(self.ys[idx])
            return x, t, y


    dataset = MyDataset(torch.FloatTensor(xs), torch.LongTensor(ts),  torch.FloatTensor(ys))

    ds_length = dataset.__len__()
    train_ds, valid_ds = torch.utils.data.random_split(dataset, [ds_length - ds_length // 10, ds_length // 10])

    train_loader = DataLoader(train_ds, batch_size=batch_size)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size)

    return train_loader, valid_loader