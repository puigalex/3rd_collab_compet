from model import Actor, Critic
from torch.optim import Adam
import torch
import numpy as np
from collections import namedtuple, deque
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LR_ACTOR = 1e-4         
LR_CRITIC = 1e-3        
WEIGHT_DECAY = 1.e-5
BUFFER_SIZE = int(1e6)
BATCH_SIZE=1024

def hard_update(target, source):
    """
    Copy network parameters from source to target
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
        

def soft_update(target, source, tau):
    """
    Perform DDPG soft update (move target params toward source based on weight
    factor tau)
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
        tau (float, 0 < x < 1): Weight factor for update
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def transpose_to_tensor(input_list):
    make_tensor = lambda x: torch.tensor(x, dtype=torch.float)
    return list(map(make_tensor, zip(*input_list)))

def transpose_list(mylist):
    return list(map(list, zip(*mylist)))

class DDPGAgent:

    def __init__(self, state_size, obs_size, action_size, num_agents):
        super(DDPGAgent, self).__init__()
        self.actor  = Actor(obs_size, action_size).to(device)
        self.critic = Critic(state_size, action_size*num_agents).to(device)
        self.target_actor = Actor(obs_size, action_size).to(device)
        self.target_critic = Critic(state_size, action_size*num_agents).to(device)
        self.noise = OUNoise(action_size, scale=1.0 )
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

    def act(self, obs, noise=0.0):
        '''Return action from the agent for a given state obs=state'''
        if type(obs) == np.ndarray:
            obs = torch.from_numpy(obs).float().to(device)
        action = self.actor(obs)
        action += noise*self.noise.noise()
        return action

    # def target_act(self, obs, noise=0.0):
    #     if type(obs) == np.ndarray:
    #         obs = torch.from_numpy(obs).float().to(device)
    #         #print(obs)
    #     action = self.target_actor(obs)
    #     action += noise*self.noise.noise()
    #     return action


class OUNoise:
    """Ornstein-Uhlenbeck process."""
    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()
    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = np.ones(self.action_dimension) * self.mu
    def noise(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return torch.tensor(self.state * self.scale).float().to(device)
    
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    #def __init__(self, action_size, buffer_size, batch_size, seed):
    def __init__(self, buffer_size = BUFFER_SIZE, batch_size = BATCH_SIZE, seed=1):        
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        #self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["obs", "state", "actions", "rewards", "next_obs", "next_state", "dones"])
        #self.seed = random.seed(seed)
        random.seed(seed)
    
    def insert(self, obs, state, actions, rewards, next_obs, next_state, dones):
        """Add a new experience to memory."""
        e = self.experience(obs, state, actions, rewards, next_obs, next_state, dones)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)        
        obs_vector      = [ np.array(e.obs) for e in experiences if e is not None]
        states          = [ np.array(e.state) for e in experiences if e is not None]
        actions_vector  = [ np.array(e.actions) for e in experiences if e is not None]
        rewards_vector  = [ np.array(e.rewards) for e in experiences if e is not None]
        next_obs_vector = [ np.array(e.next_obs) for e in experiences if e is not None]
        next_states     = [ np.array(e.next_state) for e in experiences if e is not None]
        dones_vector    = [ np.array(e.dones) for e in experiences if e is not None]
        return (obs_vector, states, actions_vector, rewards_vector, next_obs_vector, next_states, dones_vector)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
