from typing import Tuple
import flappy_bird_gym

import time
import numpy as np
from neural_net import NeuralNet

############################################
###     Config                           ###
############################################
FPS = 30

env = flappy_bird_gym.make('FlappyBird-v0')
sleep_time = 1 / FPS

############################################
###     Custom Fitness                   ###
############################################
class Reward:
    def __init__(self):
        self.reward = 0
    
    def update(self, obs, done):
        return self.abs_ypipe(obs, done)

    def abs_ypipe(self, obs, done):
        '''+ 0.1 for every frame it stays alive
        + min(0, 1 - abs(y)), y = vertical distance to next pipe'''
        self.reward += 0.1
        if done:
            self.reward += min(0, 1 - abs(obs[1]))
    
    def __str__(self):
        return f'{self.reward:.3f}'


############################################
###     Simulate                         ###
############################################
def simulate(get_action: callable, Reward=Reward, graphics=False) -> Tuple[int]:
    obs = env.reset()
    reward = Reward()
    while True:
        action = get_action(obs)

        obs, _, done, info = env.step(action)
        reward.update(obs, done)

        if graphics:
            env.render()
            time.sleep(sleep_time)
        
        if done:
            return reward.reward, info['score']


############################################
###     Run Agent                        ###
############################################
if __name__ == '__main__':
    simulate(lambda _: 0, graphics=True)
