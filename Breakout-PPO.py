import copy
import random
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import wandb
import datetime
import os
from torch.utils.data import TensorDataset, DataLoader
# from stable_baselines3.common.atari_wrappers import (  # isort:skip
#     ClipRewardEnv,
#     EpisodicLifeEnv,
#     FireResetEnv,
#     MaxAndSkipEnv,
#     NoopResetEnv,
# )

class Model(nn.Module):
    # def __init__(self, n_stack, n_actions):
    #     super(Model, self).__init__()
    #     self.conv1 = nn.Conv2d(n_stack, 16, kernel_size=8, stride=4)
    #     self.relu1 = nn.ReLU()
    #     self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
    #     self.relu2 = nn.ReLU()
    #     self.flatten = nn.Flatten()
    #     self.linear = nn.Linear(9*9*32, 256)
    #     self.relu3 = nn.ReLU()
    #     self.value_linear = nn.Linear(256, 1)
    #     self.policy_linear = nn.Linear(256, n_actions)
    #     self.log_softmax = nn.LogSoftmax(dim=1)
    #
    # def forward(self, x):
    #     x = self.relu1(self.conv1(x))
    #     x = self.relu2(self.conv2(x))
    #     x = self.flatten(x)
    #     x = self.relu3(self.linear(x))
    #     value_out = self.value_linear(x)
    #     policy_out = self.log_softmax(self.policy_linear(x))
    #     return value_out, policy_out

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
        self.value_linear = nn.Linear(512, 1)
        self.policy_linear = nn.Linear(512, n_actions)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.flatten(x)
        x = self.relu4(self.linear1(x))
        value_out = self.value_linear(x)
        policy_out = self.log_softmax(self.policy_linear(x))
        return value_out, policy_out


def init_weights(m):
    if (type(m) == nn.Linear) | (type(m) == nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight)


def init_net(net):
    for name, module in net.named_modules():
        if type(module) == nn.Linear or type(module) == nn.Conv2d:
            if name == 'value_linear':
                torch.nn.init.orthogonal_(module.weight, 1)
            elif name == 'policy_linear':
                torch.nn.init.orthogonal_(module.weight, 0.01)
            else:
                torch.nn.init.orthogonal_(module.weight, np.sqrt(2))
            torch.nn.init.constant_(module.bias, 0.0)


def preprocess(frame):
    frame = frame[:, 5:193, 8:152]
    frame = np.array([cv2.resize(frame[i], (84, 84)) / 255 for i in range(frame.shape[0])])
    return frame


def env_func(gym_id, s):
    def f():
        env = gym.make(gym_id, render_mode='rgb_array', obs_type='grayscale')
        # env = MaxAndSkipEnv(env, skip=4)   # used in "BreakoutNoFrameskip-v4"
        env.seed(s)
        env.action_space.seed(s)
        env.observation_space.seed(s)
        return env
    return f


if __name__ == '__main__':
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    n_envs = 8
    # gym_id = "BreakoutNoFrameskip-v4"
    gym_id = "ALE/Breakout-v5"
    envs_func = [env_func(gym_id, seed+i) for i in range(n_envs)]
    envs = gym.vector.AsyncVectorEnv(envs_func)
    envs.reset()
    n_actions = envs.single_action_space.n
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_stack = 4
    gamma = 0.99
    lamda = 0.95
    lr = 0.00025
    entropy_coef = 0.01
    value_loss_coef = 0.5
    n_epochs = 4  # number of epoch to update per episode
    clip_para = 0.1
    batch_size = 256
    n_steps = 128  # the number of steps to run for each iteration (the total frames is n_envs * n_step,
                  # and then run n_epoch , every epoch run multiple mini-batch,
    n_iterations = 1000000
    lr_stop_iter = 40000  # lr annealing from iter_num 0 to lr_stop_iter, after this lr_stop_iter, lr will be 0
                          # PPO paper actually is about 40000

    training_net = Model(n_stack, n_actions).to(device)
    training_net.apply(init_weights)
    # init_net(training_net)
    value_loss = nn.HuberLoss()
    optimizer = torch.optim.Adam(training_net.parameters(), lr=lr)
    score_list = []
    loss_list = []
    valueloss_list = []
    policyloss_list = []
    entropy_list = []
    eval_score_list = []
    accFrameNum_list = []
    iter_num_start = 0

    init_from = 'scratch'
    project_name = 'Breakout-PPO'
    ckpt_dir_load = './kaggle/working/'
    if not os.path.exists(ckpt_dir_load):
        ckpt_dir_load = '/kaggle/input/breakout-ckpt/'
    ckpt_dir_save = './kaggle/working/'
    if not os.path.exists(ckpt_dir_save):
        ckpt_dir_save = '/kaggle/working/'
    ckpt_file = 'ckpt_run-xx.pt'
    wandb.login(key=os.environ.get('WANDB_KEY'))
    config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
    wandb_config = {k: globals()[k] for k in config_keys}
    if init_from == 'scratch':
        run = wandb.init(project=project_name, name=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                         config=wandb_config, save_code=True)
        ckpt_file = f'ckpt_run-{run.id}.pt'
    elif init_from == 'resume':
        checkpoint = torch.load(os.path.join(ckpt_dir_load, ckpt_file), map_location=device)
        iter_num_start = checkpoint['iter_num_done'] + 1
        training_net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        loss_list = checkpoint['loss']
        score_list = checkpoint['score']
        valueloss_list = checkpoint['value_loss']
        policyloss_list = checkpoint['policy_loss']
        entropy_list = checkpoint['entropy']
        eval_score_list = checkpoint['eval_score']
        accFrameNum_list = checkpoint['accFrameNum']
        run = wandb.init(project=project_name, id=checkpoint['wandb_run_id'], resume='must', save_code=True,
                         notes=f'resume from iter {iter_num_start}')

    states_list = []
    for _ in range(n_stack):
        states, rewards, terminations, truncations, infos = envs.step(envs.action_space.sample())
        states_list.append(torch.tensor(preprocess(states), dtype=torch.float32, device=device))
    epi_return = np.zeros(n_envs)

    s = torch.zeros((n_steps, n_envs, n_stack, 84, 84), dtype=torch.float32, device=device)
    a = torch.zeros((n_steps, n_envs), dtype=torch.int64, device=device)
    r = torch.zeros((n_steps, n_envs), dtype=torch.float32, device=device)
    dones = torch.zeros((n_steps, n_envs), dtype=torch.float32, device=device)
    v = torch.zeros((n_steps, n_envs), dtype=torch.float32, device=device)
    p = torch.zeros((n_steps, n_envs, n_actions), dtype=torch.float32, device=device)

    for iter_num in range(iter_num_start, n_iterations):
        states_list = states_list[-n_stack:]
        for step in range(n_steps):
            s_stack = torch.stack(states_list[-n_stack:], dim=0).transpose(0, 1)  # (n_envs, n_stack, 84, 84)
            s[step] = s_stack
            with torch.no_grad():
                values, policies = training_net(s_stack)  # values: (n_envs, 1), policies: (n_envs, n_actions)
                actions = torch.multinomial(torch.exp(policies), num_samples=1)  # actions: (n_envs, 1)
            states, rewards, terminations, truncations, infos = envs.step(actions)
            states_list.append(torch.tensor(preprocess(states), dtype=torch.float32, device=device))
            a[step] = actions.T
            r[step] = torch.tensor(rewards, dtype=torch.float32, device=device).reshape(1, -1)
            term = np.logical_or(terminations, truncations)
            dones[step] = torch.tensor(term, dtype=torch.float32, device=device).reshape(1, -1)
            v[step] = values.T
            p[step] = policies
            epi_return += rewards
            for idx in range(n_envs):
                if term[idx]:
                    score_list.append(epi_return[idx])
                    epi_return[idx] = 0
                    wandb.log({'score': score_list[-1]})

        adv = torch.zeros((n_steps, n_envs), dtype=torch.float32, device=device)
        adv_old = torch.zeros((1, n_envs), dtype=torch.float32, device=device)
        for t in reversed(range(n_steps)):
            if t == n_steps-1:
                s_stack = torch.stack(states_list[-n_stack:], dim=0).transpose(0, 1)
                with torch.no_grad():
                    values, _ = training_net(s_stack)
                delta = r[t] + gamma * values.T * (1-dones[t]) - v[t]
            else:
                delta = r[t] + gamma * v[t+1] * (1-dones[t]) - v[t]
            adv[t] = delta + gamma * lamda * adv_old * (1-dones[t])
            adv_old = adv[t]
        r = adv + v

        ss = s.reshape(n_steps*n_envs, n_stack, 84, 84)
        aa = a.reshape(-1)
        rr = r.reshape(-1, 1)
        adv = adv.reshape(-1)
        pp = p.reshape(n_steps*n_envs, n_actions)

        a_mask = F.one_hot(aa, n_actions).to(bool)
        log_pi_old = torch.masked_select(pp, a_mask)

        dataset = TensorDataset(ss, rr, adv, log_pi_old, a_mask)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        lr_cur = (1 - iter_num/lr_stop_iter) * lr
        optimizer.param_groups[0]['lr'] = lr_cur

        training_net.train()
        for j in range(n_epochs):
            for s_batch, r_batch, adv_batch, log_pi_old_batch, a_mask_batch in dataloader:
                value_batch, policy_batch = training_net(s_batch)

                vl = value_loss(value_batch, r_batch)
                ent = - (torch.sum(policy_batch * torch.exp(policy_batch), dim=1)).mean()

                log_pi = torch.masked_select(policy_batch, a_mask_batch)
                ratio = torch.exp(log_pi - log_pi_old_batch)
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1-clip_para, 1+clip_para) * adv_batch
                pl = (-torch.min(surr1, surr2)).mean()

                l = value_loss_coef * vl + pl - entropy_coef * ent
                optimizer.zero_grad()
                l.backward()
                nn.utils.clip_grad_norm_(training_net.parameters(), 0.5)
                optimizer.step()

                loss_list.append(l.detach().cpu())
                valueloss_list.append(vl.detach().cpu())
                policyloss_list.append(pl.detach().cpu())
                entropy_list.append(ent.detach().cpu())

        accFrameNum_list.append(n_envs * n_steps * (iter_num+1))
        del ss, aa, rr, pp

        wandb.log({'loss': loss_list[-1], 'iter_num': iter_num,
                   'value_loss': valueloss_list[-1], 'policy_loss': policyloss_list[-1], 'entropy': entropy_list[-1],
                   'accFrameNum': accFrameNum_list[-1]})

        if iter_num != 0 and iter_num % 300 == 0:
            checkpoint = {'model': training_net.state_dict(), 'optimizer': optimizer.state_dict(),
                          'iter_num_done': iter_num, 'loss': loss_list, 'score': score_list,
                          'value_loss': valueloss_list, 'policy_loss': policyloss_list, 'entropy': entropy_list,
                          'eval_score': eval_score_list, 'accFrameNum': accFrameNum_list,
                          'wandb_run_id': run.id, 'wandb_config': wandb_config}
            torch.save(checkpoint, os.path.join(ckpt_dir_save, ckpt_file))

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
