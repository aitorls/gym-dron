Hay dos opciones para un entorno gym
1. Crear un paquete pip. Es lo que contine el directorio
#https://qastack.mx/programming/45068568/how-to-create-a-new-gym-environment-in-openai
#https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e
#https://github.com/openai/gym/blob/master/docs/creating-environments.md

2. Crear una clase con los siguintes métodos:
#https://stable-baselines.readthedocs.io/en/master/guide/custom_env.html

import gym
from gym import spaces

class CustomEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, arg1, arg2, ...):
    super(CustomEnv, self).__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions:
    self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
    # Example for using image as input:
    self.observation_space = spaces.Box(low=0, high=255,
                                        shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)

  def step(self, action):
    ...
    return observation, reward, done, info
  def reset(self):
    ...
    return observation  # reward, done, info can't be included
  def render(self, mode='human'):
    ...
  def close (self):
    ...
