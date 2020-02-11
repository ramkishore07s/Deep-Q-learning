#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import copy
import _pickle as pickle
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


import gym


# In[3]:


def preprocess_img(img):
    """
    Returns 80x80 image without the score at the top.
    """
    return np.mean(img, axis=2).astype(np.uint8)[::2, ::2]


# In[4]:


class RingBuffer:
    def __init__(self, size=1000000):
        self.data = [None] * (size + 1)
        self.start = 0
        self.end = 0
        
    def append(self, element):
        self.data[self.end] = element
        self.end = (self.end + 1) % len(self.data)
        if self.end == self.start:
            self.start = (self.start + 1) % len(self.data)
        
    def __getitem__(self, idx):
        return self.data[(self.start + idx) % len(self.data)]
    
    def __len__(self):
        if self.end < self.start:
            return self.end + len(self.data) - self.start
        else:
            return self.end - self.start
        
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
            
    def sample(self, n=20):
        l = len(self)
        return [self[int(np.random.uniform(0, 1) * l)] for _ in range(n)]


# In[5]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# In[6]:


torch.cuda.set_device(1)


# In[7]:


class DQN(nn.Module):

    def __init__(self, h=105, w=80, outputs=4):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=(10, 8), stride=4)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(32)

        def outsize(size, kernel_size = 4, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        cw = outsize(outsize(outsize(w)))
        ch = outsize(outsize(outsize(h)))
        head_size = cw * ch * 32
        self.head = nn.Linear(head_size, 512)
        self.head2 = nn.Linear(512, outputs)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.head(x.view(x.size(0), -1)))
        return self.head2(x.view(x.size(0), -1))


# * With probability ε select a random action at
# ** otherwise select at = maxa Q∗(φ(st), a; θ)
# ** Execute action at in emulator and observe reward rt and image xt+1 Set st+1 = st, at, xt+1 and preprocess φt+1 = φ(st+1)
# ** Store transition (φt, at, rt, φt+1) in D
# ** Sample random minibatch of transitions (φj , aj , rj , φj +1 ) from D
# ** 􏰃 rj for terminal φj+1
# ** Set yj = rj + γ maxa′ Q(φj+1, a′; θ) for non-terminal φj+1

# In[ ]:


try: os.mkdir('models/')
except: pass


# In[ ]:


def train(dqn, target, env, max_epochs=200e5, replay_buf=RingBuffer(),
          test_every=1000, test_set=None, target_update_delay=10e3, gamma=0.75,
          save_every=3e4, recent=False):
    
    epoch = 0
    episode = 0
    epsilon, diff, diff2 = 1.0, (1.0 - 0.1)/10e5, (0.1 - 0.01)/10e5
    scores = []
    huber = nn.SmoothL1Loss()
    optimizer = optim.Adam(dqn.parameters(), lr=0.0000625)
    total_reward = 0
    episode_score = 0
    episode_scores = []

    #for episode in range(no_episodes):
    while epoch < max_epochs:
        lives = 5
        episode_scores.append(episode_score)
        episode_score = 0
        episode = episode + 1
        input_buf = []
        frame = env.reset()
        is_done = False
        for _ in range(4): input_buf.append(preprocess_img(frame))
            
        while not is_done:
            term = False
            
            if epoch % target_update_delay == 0:
                target.load_state_dict(dqn.state_dict())
                
            optimizer.zero_grad()
            dqn.zero_grad()

            if np.random.uniform(0, 1) < epsilon: 
                action = env.action_space.sample()
            else:
                input = torch.cuda.FloatTensor(np.array([input_buf])/256)
                action = torch.argmax(dqn(input)).cpu().numpy()

            next_input_buf = []
            reward = 0
            for _ in range(4): 
                frame, r, is_done, life = env.step(action)
                next_input_buf.append(preprocess_img(frame))
                r = np.sign(r)
                if lives > life['ale.lives']: term, lives = True, life['ale.lives']
                reward += r
                episode_score += r
                total_reward += r
                

            replay_buf.append([np.array(input_buf), action, reward, term, np.array(next_input_buf)])
            
            if recent:
                sampled_replay = replay_buf.sample(31)
                states = [sampled_replay[i][0] for i in range(31)]
                states.append(input_buf)
                actions = [[sampled_replay[i][1]] for i in range(31)]
                actions.append([action])
                rewards = [sampled_replay[i][2] for i in range(31)]
                rewards.append(reward)
                non_terminal = [not sampled_replay[i][3] for i in range(31)]
                non_terminal.append(not term)
                next_states = [sampled_replay[i][4] for i in range(31)]
                next_states.append(input_buf)
            else:
                sampled_replay = replay_buf.sample(32)
                states = [sampled_replay[i][0] for i in range(32)]
                actions = [[sampled_replay[i][1]] for i in range(32)]
                rewards = [sampled_replay[i][2] for i in range(32)]
                non_terminal = [not sampled_replay[i][3] for i in range(32)]
                next_states = [sampled_replay[i][4] for i in range(32)]

            next_states_mask = torch.cuda.FloatTensor(non_terminal)
            
            next_state_qs = torch.max(target(torch.cuda.FloatTensor(next_states)), dim=-1)[0].detach() * next_states_mask

            if len(replay_buf) > 50000:
                output_mask = torch.zeros(32, 4).cuda()
                for i, action in enumerate(actions): output_mask[i][action] = 1

                outputs = dqn(torch.cuda.FloatTensor(np.array(states)/256))
                predicted_qs = torch.sum(output_mask * outputs, dim=1)
                actual_qs = torch.cuda.FloatTensor(rewards) + gamma * next_state_qs

                loss = huber(predicted_qs, actual_qs)
                loss.backward()

                for param in dqn.parameters():
                    param.grad.data.clamp_(-1, 1)
                optimizer.step()
            
                print(episode, epoch, str(epsilon)[:5], lives, str(sum(episode_scores[-100:])/100)[:7], episode_score, action[0], loss.detach().cpu().numpy(), end='\r')
                
                if epoch <= 105e4: epsilon = epsilon - diff
                elif epoch <= 205e4: epsilon = epsilon - diff2
            else:
                print('init: ' + str(len(replay_buf)), end='\r')
            
            input_buf = next_input_buf
            epoch = epoch + 1
        
            if epoch % test_every == 1:
                if test_set is not None:
                    with torch.no_grad():
                        scores.append(np.mean(dqn(torch.cuda.FloatTensor(test_set)).cpu().numpy()))

            if epoch % save_every == 1:
                print('')
                print("saving model: ", epoch)
                torch.save(dqn.state_dict(), "models/dqn_nr2_" + str(epoch) + ".pkl")
                pickle.dump(scores, open('models/scores_nr2.pkl', 'wb+'))
                pickle.dump(episode_scores, open('models/episode_scores_nr2.pkl', 'wb+'))
                #pickle.dump(replay_buf, open('/scratch/ramki/models/episode_replay_nr1.pkl', 'wb+'))

    return scores


# In[ ]:


#dqn = DQN()
#target = DQN()
#target.cuda()
#print(dqn.cuda())

# In[ ]:


def get_test_set():
    env = gym.make('BreakoutDeterministic-v4')

    test_set = []
    frame = env.reset()

    input_buf = []
    for _ in range(4): input_buf.append(preprocess_img(frame))
    test_set.append(input_buf)

    is_done = False
    for _ in range(100):
        action = env.action_space.sample()
        input_buf = []
        for _ in range(4):
            frame, reward, is_done, _ = env.step(action)
            input_buf.append(preprocess_img(frame))
        test_set.append(input_buf)
    return test_set

env = gym.make('BreakoutDeterministic-v4')


# In[ ]:


#scores = train(dqn, target, env, test_set=get_test_set(), replay_buf=RingBuffer(), recent=True)


# In[ ]:

def test_model(dqn, save_as='project.mp4'):
    env = gym.make('BreakoutDeterministic-v4')

    frames = []
    score = 0
    i = 0
    input_buf = []

    with torch.no_grad():
        frame = env.reset()
        frame, reward, is_done, _ = env.step(1)
        frames.append(frame)
        is_done = False
        for _ in range(4): input_buf.append(preprocess_img(frame))

        while not is_done:
            next_input_buf = []
            i = i + 1
            if np.random.uniform(0, 1) > 0.05:
                pred = dqn(torch.cuda.FloatTensor([input_buf])).cpu().numpy()
                action = np.argmax(pred)
            else:
                action = 1
            for _ in range(4):
                frame, reward, is_done, _ = env.step(action)
                frames.append(frame)
                score = score + reward
                next_input_buf.append(preprocess_img(frame))

            input_buf = next_input_buf
            
    import cv2

    img_array = []
    for img in frames:
        img = img
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img) 

    out = cv2.VideoWriter(save_as ,cv2.VideoWriter_fourcc(*'MP4V'), 30, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    return frames, score

# In[ ]:


#print(test_model(dqn))

