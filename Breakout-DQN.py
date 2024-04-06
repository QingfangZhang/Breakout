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
        self.conv1 = nn.Conv2d(n_stack, 32, kernel_size=8, stride=4)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.relu3 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(7*7*64, 512)
        self.relu4 = nn.ReLU()
        self.linear2 = nn.Linear(512, n_actions)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.flatten(x)
        x = self.relu4(self.linear1(x))
        out = self.linear2(x)
        return out


def init_weights(m):
    if (type(m) == nn.Linear) | (type(m) == nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)


class ReplayBuffer:
    def __init__(self, maxlen, n_stack):
        self.capacity = maxlen
        self.n_stack = n_stack
        self.frames = None
        self.actions = torch.empty([self.capacity], dtype=torch.uint8, device=device)
        self.rewards = torch.empty([self.capacity], dtype=torch.float32, device=device)
        self.dones = torch.empty([self.capacity], dtype=torch.bool, device=device)
        self.nums_in_buffer = 0
        self.next_idx = 0

    def store_frame(self, frame):
        # preprocessing
        frame = frame[5:193, 8:152]
        frame = cv2.resize(frame, (84, 84))
        if self.frames is None:
            self.frames = torch.empty([self.capacity] + list(frame.shape), dtype=torch.uint8, device=device)
        self.frames[self.next_idx] = torch.tensor(frame)
        ret = self.next_idx
        self.next_idx = (self.next_idx + 1) % self.capacity
        self.nums_in_buffer = min(self.capacity, self.nums_in_buffer+1)
        return ret

    def store_ard(self, idx, action, reward, done):
        self.actions[idx] = torch.tensor(action)
        self.rewards[idx] = torch.tensor(reward)
        self.dones[idx] = torch.tensor(done)

    def generate_recent_state(self):
        idx = (self.next_idx - 1) % self.capacity
        return self.generate_state(idx).unsqueeze(0).to(torch.float32)

    def generate_state(self, idx):
        """
        there are several special conditions:
        1. enough data, but start_idx < 0
           start_idx will always be < 0, and append the data frame by frame, so the frame in the tail can be taken
        2. not enough data, and start_idx < 0
           start_idx = 0, missing_frame_num will be > 0, the frame at start_idx will be duplicated for missing_frame
        3. done signal in the middle of stack
           start_idx will be set to be the next frame after done signal. the missing_frame_num will be > 0, the frame at
           start_idx will be duplicated for missing_frame

        """
        start_idx = idx - self.n_stack + 1
        # if there is not enough data in buffer
        if start_idx < 0 and self.nums_in_buffer != self.capacity:
            start_idx = 0
        for i in range(start_idx, idx):
            # include the condition that start_idx < 0. It is OK for done in the last frame idx, so exclude idx
            if self.dones[i % self.capacity]:
                start_idx = i + 1   # to separate the old episode and new episode

        missing_frame_num = self.n_stack - (idx+1 - start_idx)
        if start_idx < 0 or missing_frame_num > 0:
            states = [self.frames[start_idx] for _ in range(missing_frame_num)]
            # for start_idx < 0, missing_frame_num may be 0, but it needs to iteratively append data
            for i in range(start_idx, idx+1):
                states.append(self.frames[i % self.capacity])
            return torch.stack(states, dim=0)
        else:
            return self.frames[start_idx:idx+1]

    def sample(self, batch_size):
        idxes = random.sample(list(range(self.nums_in_buffer-1)), batch_size)
        # the last frame can not be chosen because there is no next-frame
        states = torch.stack([self.generate_state(idx).to(torch.float32) for idx in idxes], dim=0)
        actions = self.actions[idxes].to(torch.int64)
        rewards = self.rewards[idxes]
        next_states = torch.stack([self.generate_state(idx+1).to(torch.float32) for idx in idxes], dim=0)
        dones = self.dones[idxes].to(torch.float32)
        return states, actions, rewards, next_states, dones


def choose_action(action):
    if steps_done < 1000000:
        epsilon = eps_start - (eps_start - eps_end) / eps_duration * steps_done
    else:
        epsilon = eps_end
    p = np.ones(n_actions) * (epsilon/n_actions)
    p[action] = 1 - (epsilon/n_actions) * (n_actions-1)
    return np.random.choice(np.arange(4), p=p)


env = gym.make('ALE/Breakout-v5', render_mode='rgb_array', obs_type='grayscale')
state, info = env.reset()
# for _ in range(100):
#     action = np.random.randint(0, 4)
#     obs, reward, terminated, truncated, info = env.step(action)
#     if terminated or truncated:
#         env.reset()
action_space = [0, 1, 2, 3]
n_actions = len(action_space)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

eps_start = 1
eps_end = 0.1
eps_duration = 1000000
steps_done = 0
training_frames = 100000000
batch_size = 32
n_stack = 4
gamma = 0.99
lr = 0.00025
learning_start_frame = 50000
min_frames = batch_size

experience = ReplayBuffer(maxlen=1000000, n_stack=n_stack)
training_net = Model(n_stack, n_actions).to(device)
training_net.apply(init_weights)
target_net = Model(n_stack, n_actions).to(device)
target_net.load_state_dict(training_net.state_dict())
loss = nn.HuberLoss()
optimizer = torch.optim.RMSprop(training_net.parameters(), lr=lr)
score_list = []
loss_list = []
step_list = []

init_from = 'scratch'
if init_from == 'resume':
    min_frames = learning_start_frame
    # because if loading checkpoint, no experience stored in ckpt, it needs to collect new experience
project_name = 'Breakout-DQN'
ckpt_dir_load = './kaggle/working/'
if not os.path.exists(ckpt_dir_load):
    ckpt_dir_load = '/kaggle/input/breakout-ckpt/'
ckpt_dir_save = './kaggle/working/'
if not os.path.exists(ckpt_dir_save):
    ckpt_dir_save = '/kaggle/working/'
ckpt_file = 'ckpt_run-lu4six2t.pt'
# with open('wandb_key.txt', 'r') as file:
#     wandb_key = file.read()
wandb.login(key='95cb6fba4e394f1779f2eb1151f12c0b77c0765b')
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
wandb_config = {k: globals()[k] for k in config_keys}
if init_from == 'scratch':
    run = wandb.init(project=project_name, name=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                     config=wandb_config, save_code=True)
    ckpt_file = f'ckpt_run-{run.id}.pt'
elif init_from == 'resume':
    checkpoint = torch.load(os.path.join(ckpt_dir_load, ckpt_file), map_location=device)
    steps_done = checkpoint['steps']
    training_net.load_state_dict(checkpoint['model'])
    target_net.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    loss_list = checkpoint['loss']
    score_list = checkpoint['score']
    # experience = checkpoint['experience']
    run = wandb.init(project=project_name, id=checkpoint['wandb_run_id'], resume='must', save_code=True,
                     notes=f'resume from iter {steps_done}')

while steps_done < training_frames:
    env.reset()
    state, reward, terminated, truncated, _ = env.step(env.action_space.sample())
    score = 0

    while not terminated and not truncated:
        frame_idx = experience.store_frame(state)
        if steps_done < learning_start_frame:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                y = training_net(experience.generate_recent_state())
                action = choose_action(torch.argmax(y, dim=1))
        steps_done += 1
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        score += reward

        experience.store_ard(frame_idx, action, reward, done)

        if steps_done > learning_start_frame and experience.nums_in_buffer > min_frames:
            s, a, r, ns, d = experience.sample(batch_size)

            training_net.train()
            a_mask = F.one_hot(a, n_actions).to(bool)
            yhat = training_net(s)
            ytemp = torch.masked_select(yhat, a_mask)
            with torch.no_grad():
                y = target_net(ns)
                target = r + gamma * torch.max(y, dim=1)[0] * (1 - d)
            l = loss(ytemp, target)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            loss_list.append(l.detach().cpu())
            step_list.append(steps_done)
            del s, a, r, ns, d, l, yhat, ytemp, target
            torch.cuda.empty_cache()

            if steps_done != 0 and steps_done % 10000 == 0:
                target_net.load_state_dict(training_net.state_dict())

                checkpoint = {'model': training_net.state_dict(), 'optimizer': optimizer.state_dict(),
                              'steps': steps_done, 'loss': loss_list, 'score': score_list,
                              # 'experience': experience,
                              'wandb_run_id': run.id, 'wandb_config': wandb_config}
                wandb.log({'score': score_list[-1], 'loss': loss_list[-1], 'steps': steps_done})
                torch.save(checkpoint, os.path.join(ckpt_dir_save, ckpt_file))

        if done:
            score_list.append(score)
            # plt.figure(1, figsize=(12, 6))
            # plt.clf()
            # plt.subplot(1, 2, 1)
            # plt.plot(score_list)
            # plt.xlabel('episode num')
            # plt.ylabel('score')
            # plt.title('score')
            # plt.subplot(1, 2, 2)
            # plt.plot(step_list[30:], loss_list[30:])
            # plt.xlabel('steps_done')
            # plt.ylabel('loss')
            # plt.title('loss')
            # plt.pause(0.001)
            break
        else:
            state = next_state
