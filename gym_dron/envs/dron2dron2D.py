import numpy as np
import random
import gym
from gym import spaces

class Dron2dron_2DC(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['console']}

  def __init__(self,gride_size = 100,max_reward = 1000,step_reward=-1, distance_colision = 2.5):
    super(Dron2dron_2DC, self).__init__()

    self.grid_size = gride_size

    #CONSTANTES
    self.MAX_REWARD = max_reward
    self.MIN_REWARD = -max_reward
    self.STEP_REWARD = step_reward
  
    self.MIN_DISTANCE_COLISION = 1.5
    self.MAX_DISTANCE_COLISION = distance_colision

    #TODO
    self.N_FEATURES = 4
    self.N_ACTIONS = 4
    self.DIM_ACTIONS = 2 # [x,y] z=2.3

    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions:
    #self.action_space = spaces.Discrete(N_ACTIONS) #Esto lo podemos cambiar a Box
    self.action_space = spaces.Box(low=-1, high=1,
                                        shape=(self.DIM_ACTIONS,), dtype=np.float32) #torch.from_numpy()
    # Example for using image as input:
    self.observation_space = spaces.Box(low=-self.grid_size, high=self.grid_size,
                                        shape=(self.N_FEATURES,), dtype=np.float32) #Mapa aereo



  def step(self, action):
    
    def constante(p):
      return p

    def dibujarRect(p,side = 12):
      if p[0] == 0. and p[1] < side:
        p += np.array([0.,1.])
      elif p[1] == side and p[0] < side:
        p += np.array([1.,0.])
      elif p[0] == side and p[1] > 0:
        p -= np.array([0.,1.])
      elif p[1] == 0 and p[0] > 0:
        p -= np.array([1.,0.])
      return p

    def step_random(p):
      num = random.randrange(2*2) 
      div = num//2
      mod = num%2
      a = np.zeros(2)
      a[mod] = (-2)*div+1

      return p+a

    other_dron_position = dibujarRect
    
    self.dron1_pos = other_dron_position(self.dron1_pos)
    
    self.dron2_pos = self.dron2_pos+action

    # Account for the boundaries of the grid
    self.dron2_pos = np.clip(self.dron2_pos, -self.grid_size, self.grid_size) #torch.clamp

    # Are we at the left of the grid?
    done = bool((self.dron2_pos == self.dron1_pos).all())

    # Null reward everywhere except when reaching the goal (left of the grid)
    def reward_inteligent(p1,p2):
      distance = np.linalg.norm(p2-p1,np.inf) #Creo que es mas inteligente que la norma 2   
      r = -distance

      if distance < self.MAX_DISTANCE_COLISION:
          r = self.MAX_REWARD
      
      return float(r)
    
    reward = reward_inteligent(self.dron2_pos,self.dron1_pos)

    # Optionally we can pass additional info, we are not using that for now
    info = {}

    return np.concatenate((self.dron1_pos, self.dron2_pos),axis=None), reward, done, info
    #return observation, reward, done, info
  
  
  def reset(self):
    """
    Important: the observation must be a 2 numpy array
    :return: (np.array,np.array) 
    """
    # Initialize the agent at the right of the grid
    self.dron1_pos = np.array([0,0]).astype(np.float32) #np
    self.dron2_pos = np.array([-5,0]).astype(np.float32)
    # here we convert to float32 to make it more general (in case we want to use continuous actions)
    return np.concatenate((self.dron1_pos, self.dron2_pos),axis=None)



  def close (self):
    pass
    
    

class Dron2dron_2DD(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['console']}

  def __init__(self,gride_size = 100,max_reward = 1000,step_reward=-1, distance_colision = 2.5):
    super(Dron2dron_2DD, self).__init__()

    self.grid_size = gride_size

    #CONSTANTES
    self.MAX_REWARD = max_reward
    self.MIN_REWARD = -max_reward
    self.STEP_REWARD = step_reward
  
    self.MIN_DISTANCE_COLISION = 1.5
    self.MAX_DISTANCE_COLISION = distance_colision

    #TODO
    self.N_FEATURES = 4
    self.N_ACTIONS = 4 # [ALANTE, ATRAS, IZQ, DER]
    self.MOVS =  np.array([[1.,0],[-1.,0],[0,1.],[0,-1.]])

    self.DIM_ACTIONS = 2 # [x,y] z=2.3

    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions:
    self.action_space = spaces.Discrete(self.N_ACTIONS) 
    #self.action_space = spaces.Box(low=-1, high=1,
    #                                    shape=(self.DIM_ACTIONS,), dtype=np.float32) #torch.from_numpy()
    # Example for using image as input:
    self.observation_space = spaces.Box(low=-self.grid_size, high=self.grid_size,
                                        shape=(self.N_FEATURES,), dtype=np.float32) #Mapa aereo



  def step(self, action):
    
    def constante(p):
      return p

    def dibujarRect(p,side = 12):
      if p[0] == 0. and p[1] < side:
        p += np.array([0.,1.])
      elif p[1] == side and p[0] < side:
        p += np.array([1.,0.])
      elif p[0] == side and p[1] > 0:
        p -= np.array([0.,1.])
      elif p[1] == 0 and p[0] > 0:
        p -= np.array([1.,0.])
      return p

    def step_random(p):
      num = random.randrange(2*2) 
      div = num//2
      mod = num%2
      a = np.zeros(2)
      a[mod] = (-2)*div+1

      return p+a

    other_dron_position = dibujarRect
    
    self.dron1_pos = other_dron_position(self.dron1_pos)
    
    self.dron2_pos = self.dron2_pos+self.MOVS[action.item()]

    # Account for the boundaries of the grid
    self.dron2_pos = np.clip(self.dron2_pos, -self.grid_size, self.grid_size) #torch.clamp

    # Are we at the left of the grid?
    done = bool((self.dron2_pos == self.dron1_pos).all())

    # Null reward everywhere except when reaching the goal (left of the grid)
    def reward_inteligent(p1,p2):
      distance = np.linalg.norm(p2-p1,np.inf) #Creo que es mas inteligente que la norma 2   
      r = -distance

      if distance < self.MAX_DISTANCE_COLISION:
          r = self.MAX_REWARD
      
      return float(r)
    
    reward = reward_inteligent(self.dron2_pos,self.dron1_pos)

    # Optionally we can pass additional info, we are not using that for now
    info = {}

    return np.concatenate((self.dron1_pos, self.dron2_pos),axis=None), reward, done, info
    #return observation, reward, done, info
  
  
  def reset(self):
    # Initialize two agent
    self.dron1_pos = np.array([0,0]).astype(np.float32) 
    self.dron2_pos = np.array([-5,0]).astype(np.float32)

    return np.concatenate((self.dron1_pos, self.dron2_pos),axis=None)



  def close (self):
    pass    
