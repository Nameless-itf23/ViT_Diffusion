import torch
import torch.nn as nn

import dataset
from model.vit import Vit
from tools import torch_short

import yaml
import datetime
import pytz


jst = pytz.timezone('Asia/Tokyo')

train_loader, valid_loader = dataset.dataloader()

with open('config/train_config.yaml') as file:
    config = yaml.safe_load(file)

batch =  config['batch']
lr =  config['lr']
epoch = config['epoch']

model = Vit()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


def save():
    current_time = datetime.datetime.now(tz=jst).strftime(r'%Y%m%d_%H%M%S')
    torch.save(model.state_dict(), f'weights/model_{current_time}.pth')

train_data, test_data = torch_short.fit(model, train_loader, valid_loader, criterion, optimizer, epoch, call=save)