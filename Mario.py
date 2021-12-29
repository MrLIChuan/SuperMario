#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install gym_super_mario_bros==7.3.0 nes_py')


# In[2]:


import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


# In[3]:


env = gym_super_mario_bros.make('SuperMarioBros-v0')
env =JoypadSpace(env, SIMPLE_MOVEMENT)


# In[34]:


#env.observation_space.shape


# In[35]:


#env.action_space


# In[36]:


#SIMPLE_MOVEMENT


# In[ ]:


done = True
#Loop through each frame in the game
for step in range(100000):
    if done:
        #start the game
        env.reset()
    state, revard, done, info = env.step(env.action_space.sample())
    env.render()
env.close()


# In[ ]:


#state = env.reset()


# In[1]:


#env.step(1)[3]


# In[4]:


#install Pytorch
#!pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio===0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# Install stable baselines for RL stuff
get_ipython().system('pip install stable-baselines3[extra]')


# In[5]:


# import frame stocker wrapper and grayscaling wrapper
from gym.wrappers import FrameStack, GrayScaleObservation

from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

from matplotlib import pyplot as plt


# In[6]:


# 1. Create the base environment
env = gym_super_mario_bros.make('SuperMarioBros-v0')
# 2. Simplify the controls 
env = JoypadSpace(env, SIMPLE_MOVEMENT)
# 3. Grayscale
env = GrayScaleObservation(env, keep_dim=True)
# 4. Wrap inside the Dummy Environment
env = DummyVecEnv([lambda: env])
# 5. Stack the frames
env = VecFrameStack(env, 4, channels_order='last')#


# In[7]:


state=env.reset()


# In[8]:


#state.shape
state, reward, done, info =  env.step([5])


# In[9]:


#use Matplolib to show the game frame
#plt.imshow(state[0])


# In[10]:


plt.figure(figsize=(20,16))
for idx in range(state.shape[3]):
    plt.subplot(1,4,idx+1)
    plt.imshow(state[0][:,:,idx])
plt.show()


# In[11]:


# Import os for file path management
import os 
# Import PPO for algos
from stable_baselines3 import PPO
# Import Base Callback for saving models
from stable_baselines3.common.callbacks import BaseCallback


# In[12]:


class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True


# In[13]:


CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'


# In[14]:


# Setup model saving callback
callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)


# In[15]:


# This is the AI model started
model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.000001, 
            n_steps=512) #MlpPolicy


# In[16]:


# Train the AI model, this is where the AI model starts to learn
model.learn(total_timesteps=1000000, callback=callback)


# In[17]:


model.save('thisisatestmodel')


# In[16]:


# Load model
model = PPO.load('./train/best_model_10000')


# In[17]:


state = env.reset()


# In[ ]:


# Start the game 
state = env.reset()
# Loop through the game
while True: 
    
    action, _ = model.predict(state)
    state, reward, done, info = env.step(action)
    env.render()


# In[ ]:




