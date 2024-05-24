import copy
import random
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
from collections import deque
import wandb
import datetime
import os


class Model(nn.Module):
    def __init__(self, n_stack, n_actions):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(n_stack, 16, kernel_size=8, stride=4)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.relu2 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(9*9*32, 256)
        self.relu3 = nn.ReLU()
        self.value_linear = nn.Linear(256, 1)
        self.policy_linear = nn.Linear(256, n_actions)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.flatten(x)
        x = self.relu3(self.linear(x))
        value_out = self.value_linear(x)
        policy_out = self.log_softmax(self.policy_linear(x))
        return value_out, policy_out


def init_weights(m):
    if (type(m) == nn.Linear) | (type(m) == nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight)


def preprocess(frame):
    frame = frame[5:193, 8:152]
    frame = cv2.resize(frame, (84, 84)) / 255
    return frame


@torch.no_grad()
def evaluate_model():
    training_net.eval()
    env.reset()
    eval_score = 0
    state, reward, terminated, truncated, _ = env.step(env.action_space.sample())
    state_list = [torch.tensor(preprocess(state), dtype=torch.float32, device=device)]
    frame_idx = 1
    while len(state_list) < n_stack:
        state, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        state_list.append(torch.tensor(preprocess(state), dtype=torch.float32, device=device))
        frame_idx += 1
    s_stack = torch.stack([x for x in state_list[(frame_idx-n_stack):frame_idx]], dim=0).unsqueeze(0)
    while not terminated and not truncated:
        _, policy = training_net(s_stack)
        action = torch.argmax(policy.reshape(-1)).item()
        state, reward, terminated, truncated, _ = env.step(action)
        state_list.append(torch.tensor(preprocess(state), dtype=torch.float32, device=device))
        frame_idx += 1
        eval_score += reward
        s_stack = torch.stack([x for x in state_list[(frame_idx-n_stack):frame_idx]], dim=0).unsqueeze(0)
    eval_score_list.append(eval_score)
    del s_stack, state_list, frame_idx, reward, terminated, truncated, action, policy, eval_score, state


env = gym.make('ALE/Breakout-v5', render_mode='rgb_array', obs_type='grayscale')
state, info = env.reset()
# while True:
#     obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
#     if terminated or truncated:
#         break
# print('terminated or truncated')
action_space = [0, 1, 2, 3]
n_actions = len(action_space)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

n_stack = 4
gamma = 0.99
lr = 0.001
episode_num = 1000000
entropy_coef = 0.01
value_loss_coef = 0.5
k = 5  # how many step of reward look ahead

training_net = Model(n_stack, n_actions).to(device)
training_net.apply(init_weights)
value_loss = nn.HuberLoss()
optimizer = torch.optim.RMSprop(training_net.parameters(), lr=lr)
score_list = []
loss_list = []
valueloss_list = []
policyloss_list = []
entropy_list = []
eval_score_list = []
epi_num_start = 0

init_from = 'resume'
project_name = 'Breakout-A2C'
ckpt_dir_load = './kaggle/working/'
if not os.path.exists(ckpt_dir_load):
    ckpt_dir_load = '/kaggle/input/breakout-ckpt/'
ckpt_dir_save = './kaggle/working/'
if not os.path.exists(ckpt_dir_save):
    ckpt_dir_save = '/kaggle/working/'
ckpt_file = 'ckpt_run-4f43rviz.pt'
wandb.login(key=os.environ.get('WANDB_KEY'))
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
wandb_config = {k: globals()[k] for k in config_keys}
if init_from == 'scratch':
    run = wandb.init(project=project_name, name=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                     config=wandb_config, save_code=True)
    ckpt_file = f'ckpt_run-{run.id}.pt'
elif init_from == 'resume':
    checkpoint = torch.load(os.path.join(ckpt_dir_load, ckpt_file), map_location=device)
    epi_num_start = checkpoint['epi_num_done'] + 1
    training_net.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    loss_list = checkpoint['loss']
    score_list = checkpoint['score']
    valueloss_list = checkpoint['value_loss']
    policyloss_list = checkpoint['policy_loss']
    entropy_list = checkpoint['entropy']
    eval_score_list = checkpoint['eval_score']
    run = wandb.init(project=project_name, id=checkpoint['wandb_run_id'], resume='must', save_code=True,
                     notes=f'resume from iter {epi_num_start}')

for epi_num in range(epi_num_start, episode_num):
    env.reset()
    state, reward, terminated, truncated, info = env.step(env.action_space.sample())
    state_list = [torch.tensor(preprocess(state), dtype=torch.float32, device=device)]
    frame_idx = 1
    while len(state_list) < n_stack:
        state, reward, terminated, truncated, info = env.step(env.action_space.sample())
        state_list.append(torch.tensor(preprocess(state), dtype=torch.float32, device=device))
        frame_idx += 1
    s_stack = torch.stack([x for x in state_list[(frame_idx-n_stack):frame_idx]], dim=0).unsqueeze(0)
    s_batch = copy.deepcopy(s_stack)
    a_batch = []
    r_batch = []
    while not terminated and not truncated:
        with torch.no_grad():
            _, policy = training_net(s_stack)
            action = torch.multinomial(torch.exp(policy).reshape(-1), num_samples=1).item()
        state, reward, terminated, truncated, info = env.step(action)
        state_list.append(torch.tensor(preprocess(state), dtype=torch.float32, device=device))
        frame_idx += 1
        s_stack = torch.stack([x for x in state_list[(frame_idx-n_stack):frame_idx]], dim=0).unsqueeze(0)
        s_batch = torch.cat((s_batch, s_stack), dim=0)
        a_batch.append(action)
        r_batch.append(reward)

    s = s_batch[:-1]
    ns = s_batch[1:]
    a = torch.tensor(a_batch, dtype=torch.int64, device=device)
    r = torch.tensor(r_batch, dtype=torch.float32, device=device).reshape(-1, 1)  # (batch_size, 1)
    score = sum(r_batch)
    gamma_list = torch.tensor([[1]], dtype=torch.float32, device=device)
    for j in range(1, k):
        gamma_list = torch.cat((gamma_list, torch.tensor([[gamma**j]], dtype=torch.float32, device=device)), dim=0)
    nr = len(r)
    for i in range(nr-k):
        r[i] = torch.matmul(gamma_list.T, r[i:i+k])
    for i in range(nr-k, nr-2):
        r[i] = torch.matmul(gamma_list[:(nr-i-1)].T, r[i:(nr-1)])

    training_net.train()
    value, policy = training_net(s)
    r[:(nr-k)] += (gamma**k) * (value[k:].detach())
    for i in range(nr-k, nr-1):
        r[i] += (gamma**(nr-i-1)) * (value[-1].detach())
    adv = r - value.detach()
    vl = value_loss(value, r)
    ent = - torch.sum(policy * torch.exp(policy), dim=1)
    a_mask = F.one_hot(a, n_actions).to(bool)
    pl = -(torch.masked_select(policy, a_mask) * (adv.reshape(-1)) + entropy_coef * ent).mean()
    l = value_loss_coef * vl + pl
    optimizer.zero_grad()
    l.backward()
    optimizer.step()

    loss_list.append(l.detach().cpu())
    valueloss_list.append(vl.detach().cpu())
    policyloss_list.append(pl.detach().cpu())
    score_list.append(score)
    entropy_list.append(ent.detach().mean().cpu())

    if epi_num != 0 and epi_num % 100 == 0:
        checkpoint = {'model': training_net.state_dict(), 'optimizer': optimizer.state_dict(),
                      'epi_num_done': epi_num, 'loss': loss_list, 'score': score_list,
                      'value_loss': valueloss_list, 'policy_loss': policyloss_list, 'entropy': entropy_list,
                      'eval_score': eval_score_list,
                      'wandb_run_id': run.id, 'wandb_config': wandb_config}
        wandb.log({'score': score_list[-1], 'loss': loss_list[-1], 'epi_num': epi_num,
                   'value_loss': valueloss_list[-1], 'policy_loss': policyloss_list[-1], 'entropy': entropy_list[-1]})
        torch.save(checkpoint, os.path.join(ckpt_dir_save, ckpt_file))

    if epi_num != 0 and epi_num % 2000 == 0:
        evaluate_model()
        wandb.log({'eval_score': eval_score_list[-1]})

    # plt.figure(1, figsize=(12, 6))
    # plt.clf()
    # plt.subplot(2, 2, 1)
    # plt.plot(score_list)
    # plt.xlabel('episode num')
    # plt.ylabel('score')
    # plt.title('score')
    # plt.subplot(2, 2, 2)
    # plt.plot(loss_list)
    # plt.xlabel('episode num')
    # plt.ylabel('loss')
    # plt.title('loss')
    # plt.subplot(2, 2, 3)
    # plt.plot(valueloss_list)
    # plt.xlabel('episode num')
    # plt.ylabel('value loss')
    # plt.subplot(2, 2, 4)
    # plt.plot(policyloss_list)
    # plt.xlabel('episode num')
    # plt.ylabel('policy loss')
    # plt.pause(0.001)
