import torch
import torch.nn as nn

import dataset
from model.vit import Vit
from tools import torch_short

import yaml
import datetime


with open('config/train_config.yaml') as file:
    config = yaml.safe_load(file)

batch = config['batch']
lr =  config['lr']
epoch = config['epoch']
data_num_rate = config['data_num_rate']

train_loader, valid_loader = dataset.dataloader(batch_size=batch, data_num_rate=data_num_rate)

model = Vit()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


def save():
    torch.save(model.state_dict(), f'weights/model.pth')

train_data, test_data = torch_short.fit(model, train_loader, valid_loader, criterion, optimizer, epoch, call=save)