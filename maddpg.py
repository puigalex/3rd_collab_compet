from ddpg import DDPGAgent
import torch
from ddpg import soft_update, transpose_to_tensor, transpose_list
import numpy as np
import pdb

DISCOUNT_FACTOR = 0.95 
TAU = 0.02
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def reshape_sample(vec3):
    ns, nm, ni = vec3.shape
    return vec3.reshape(nm, ns, ni)

def transpose_to_tensor(input_list):
    make_tensor = lambda x: torch.tensor(x, device = device, dtype=torch.float)
    return list(map(make_tensor, zip(*input_list)))

class MADDPG:
    #def __init__(self, agents_archs):
    def __init__(self, state_size, obs_size, action_size, num_agents):
        super(MADDPG, self).__init__()

        self.maddpg_agent = [DDPGAgent(state_size, obs_size, action_size, num_agents) for x in range(num_agents)]
        self.discount_factor = DISCOUNT_FACTOR
        self.tau = TAU
        self.iter = 0

    def get_actors(self):
        '''retreive all agent's actors from MAGGPs Objet'''
        actors = [ddpg_agent.actor for ddpg_agent in self.maddpg_agent]
        return actors

    def get_target_actors(self):
        '''retreive all agent's target actors from MAGGPs Objet'''
        target_actors = [ddpg_agent.target_actor for ddpg_agent in self.maddpg_agent]
        return target_actors

    def get_critics(self):
        '''retreive all agent's critics from MAGGPs Objet'''
        actors = [ddpg_agent.critic for ddpg_agent in self.maddpg_agent]
        return actors

    def act(self, obs_all_agents, noise=0.0):
        '''retreive actions from the agents'''
        actions = [agent.act(obs, noise) for agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return actions

    # def target_act(self, obs_all_agents, noise=0.0):
    #     '''retreive target nteworks actions'''
    #     target_actions = [ddpg_agent.target_act(obs, noise) for ddpg_agent, obs in zip(self.maddpg_agent, obs_all_agents)]
    #     return target_actions
    

    def update(self, samples, agent_number):
        '''update the actor and the critic for the agents'''
        obs, obs_full, action, reward, next_obs, next_obs_full, done = map(transpose_to_tensor, samples)
        obs_full = torch.stack(obs_full)
        next_obs_full = torch.stack(next_obs_full)
        
        agent = self.maddpg_agent[agent_number]
        agent.critic_optimizer.zero_grad()
        # target_actions = self.target_act(next_obs)
        target_actions = self.act(next_obs)
        target_actions = torch.cat(target_actions, dim=1)

        with torch.no_grad():
            q_next = agent.target_critic(next_obs_full.t(), target_actions)
        
        y = reward[agent_number].view(-1, 1) + self.discount_factor * q_next * (1 - done[agent_number].view(-1, 1))
        action = torch.cat(action, dim=1)
        #print(action)
        q = agent.critic(obs_full.t(), action)
        huber_loss = torch.nn.SmoothL1Loss()
        critic_loss = huber_loss(q, y.detach())
        critic_loss.backward()
        agent.critic_optimizer.step()
        agent.actor_optimizer.zero_grad()
        q_input = [ self.maddpg_agent[i].actor(ob) if i == agent_number \
                   else self.maddpg_agent[i].actor(ob).detach()
                   for i, ob in enumerate(obs) ]
                
        q_input = torch.cat(q_input, dim=1)
        actor_loss = -agent.critic(obs_full.t(), q_input).mean()
        actor_loss.backward()
        agent.actor_optimizer.step()

    def soft_update_targets(self):
        '''perform soft update to targets'''
        self.iter += 1
        for ddpg_agent in self.maddpg_agent:
            soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.tau)
            soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.tau)
            
            
            




