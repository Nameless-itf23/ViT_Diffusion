import torch
import numpy as np
import matplotlib.pyplot as plt
from model.vit import Vit

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

model = Vit()

model.load_state_dict(torch.load('weights/model.pth', map_location=torch.device(device)))

def noise_scheduler(t):  # t -> sigma
    return 1 - t / T

def make_noise():
    return np.random.normal(0.5, SIGMA, (BATCH_SIZE, 3, HEIGHT, WIDTH))

def show(out):
    out = np.transpose(out, (0, 2, 3, 1))
    plt.imshow(out[0])
    plt.show()

T = 500

SAMPLING_STEPS = 1
WIDTH = 32
HEIGHT = 32
BATCH_SIZE = 1
BATCH_COUNT = 1

SIGMA = 0.3
STRENGTH = 350

for _ in range(BATCH_COUNT):

    # ノイズを用意
    noise = make_noise()
    noise = torch.from_numpy(noise.astype(np.float32))

    # デノイズ
    for t in reversed(range(T - 1)):
        noise = model(noise, torch.tensor([[t]] * BATCH_SIZE))

    # ノイズ付加 -> デノイズの繰り返し
    for _ in range(SAMPLING_STEPS - 1):

        # ノイズを用意
        noise = noise.detach().numpy()
        noise = np.random.normal(0, noise_scheduler(t), (BATCH_SIZE, 3, HEIGHT, WIDTH))
        noise = noise_scheduler(STRENGTH) * noise + (1 - noise_scheduler(STRENGTH)) * make_noise()
        noise = torch.from_numpy(noise.astype(np.float32))

        # デノイズ
        for t in reversed(range(STRENGTH)):
            noise = model(noise, torch.tensor([[t]] * BATCH_SIZE))

    # 結果を保存
    output = noise.detach().numpy().clip(0, 1)

show(output)