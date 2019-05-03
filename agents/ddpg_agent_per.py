import numpy as np
import random
import copy
from collections import namedtuple, deque

from agents.model import Actor, Critic
from agents.bst import FixedSize_BinarySearchTree
from agents.SumTree import SumTree

import torch
import torch.nn.functional as F
import torch.optim as optim

WEIGHT_DECAY = 0        # L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self,state_size, action_size,single_rotor_control=False, buffer_size=int(1e5), batch_size=128, 
                    gamma=0.98, tau=1e-3, lr_actor=1e-4,lr_critic=1e-3, random_seed=0):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size

        # rotor control mode
        self.single_rotor_control = single_rotor_control

        if self.single_rotor_control:
            action_size = 1

        # define hyperparameters
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic

        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr_critic, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(size=action_size, seed=random_seed)

        # Replay memory
        self.memory = ReplayBuffer(buffer_size, random_seed)

        # counter for time steps
        self.time_step = 0

        #self.soft_update(self.critic_local, self.critic_target, 1.0)
        #self.soft_update(self.actor_local, self.actor_target, 1.0)
    
    def step(self,state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        
        # Save experience / reward
        if self.single_rotor_control:
            self.memory.add(state, action[0], reward, next_state, done)
        else:
            self.memory.add(state, action, reward, next_state, done)
        
        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences, self.gamma)

        # increase time step count
        self.time_step+=1

    def act(self, states, add_noise=True):
        """Returns actions for given state as per current policy."""
        states = torch.from_numpy(states).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(states.unsqueeze(0)).cpu().data.numpy()[0]
        self.actor_local.train()
        if add_noise:
            #noise = np.repeat(,self.action_size)
            actions += self.noise.sample()

        if self.single_rotor_control:
            actions = np.repeat(actions,self.action_size)

        return np.clip(actions, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        
        states, actions, rewards, next_states, dones, idxs, is_weights = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute Q value predictions
        Q_expected = self.critic_local(states, actions)

        # compute td error
        td_error = Q_targets - Q_expected
        # update td error in Replay buffer
        self.memory.update_priorities(idxs,td_error.detach().cpu().numpy().squeeze())

        # compute critic loss
        critic_loss = ((is_weights*td_error)**2).mean()

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -(is_weights * self.critic_local(states, actions_pred).squeeze()).mean()

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class GaussianNoise:
    '''Gaussian Noise'''

    def __init__(self,size,seed,sigma_start=1.0,sigma_decay=0.9999,sigma_end=0.01):
        """Initialize parameters and noise process."""
        self.size = size
        self.seed = random.seed(seed)

        self.sigma = sigma_start
        self.sigma_decay = sigma_decay
        self.sigma_end = sigma_end

    def reset(self):
        pass

    def sample(self):
        # sample noise
        noise = self.sigma * np.random.standard_normal(self.size)

        # decay sigma
        self.sigma = max(self.sigma_decay*self.sigma,self.sigma_end)
        return noise

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma_start=1.0,sigma_decay=0.9999,sigma_end=1e-3):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.size = size
        self.seed = random.seed(seed)
        self.reset()

        self.sigma = sigma_start
        self.sigma_decay = sigma_decay
        self.sigma_end = sigma_end

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)#np.array([random.random() for i in range(len(x))])
        self.state = x + dx

        self.sigma = max(self.sigma_decay*self.sigma,self.sigma_end)
        return self.state


class ReplayBuffer:  # stored as ( s, a, r, s_ ) in SumTree
    def __init__(self, buffer_size, seed, alpha=0.4, beta=0.4):
        self.sum_tree = SumTree(capacity=buffer_size)
        self.max_tree = FixedSize_BinarySearchTree(capacity=buffer_size)
        self.capacity = buffer_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.epsilon = 1e-5
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = 1e-3
        self.seed = random.seed(seed)
        self.base_priority = self.epsilon**self.alpha

    def _get_priority(self, error):
        return (error + self.epsilon) ** self.alpha

    def _get_max_priority(self):
        try:
            max_priority = self.max_tree.max_value()
        except:
            max_priority = self.base_priority

        return max_priority

    def add(self,state,action,reward,next_state,done):
        max_priority = self._get_max_priority()
        e = self.experience(state,action,reward,next_state,done)
        self.sum_tree.add(max_priority, e)
        self.max_tree.add(max_priority)

    def sample(self, batch_size):
        experiences = []
        idxs = []
        segment = self.sum_tree.total() / batch_size
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.sum_tree.get(s)
            priorities.append(p)
            experiences.append(data)
            idxs.append(idx)

        sampling_probabilities = np.array(priorities) / self.sum_tree.total()
        is_weights = np.power(self.sum_tree.n_entries * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()
        is_weights = torch.from_numpy(np.vstack(is_weights)).float().to(device)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return states,actions,rewards,next_states,dones, idxs, is_weights

    def update_priorities(self, idxs, td_error):
        abs_td_errors = np.abs(td_error)
        for idx,error in zip(idxs,abs_td_errors):
            p = self._get_priority(error)
            self.sum_tree.update(idx, p)
            self.max_tree.update(p,idx-self.capacity+1)

    def __len__(self):
        return self.sum_tree.n_entries

# class ReplayBuffer:
#     def __init__(self,buffer_size,seed,alpha=0.6,beta=0.4):
#         self.buffer = deque(maxlen=buffer_size)
#         self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
#         self.tree = FixedSize_BinarySearchTree(capacity=buffer_size)
#         self.epsilon = 1e-5
#         self.alpha = alpha
#         self.beta = beta
#         self.beta_increment_per_sampling = 1e-3
#         self.base_priority = self.epsilon**self.alpha

#     def add(self,state,action,reward,next_state,done):
        
#         max_priority = self._get_max_priority() 
#         self.tree.add(max_priority)    

#         e = self.experience(state,action,reward,next_state,done)
#         self.buffer.append(e)
    
#     def _get_max_priority(self):
#         try:
#             max_priority = self.tree.max_value()
#         except:
#             max_priority = self.base_priority

#         return max_priority

#     def update_priorities(self,idxs,td_errors):
#         new_priorities = np.abs(td_errors)**self.alpha

#         #print ("update: {:.2f},{:.2f},{:.2f}".format(self.tree.value_sum,np.max(self.tree.values),np.max(new_priorities)))
#         for idx,new_priority in zip(idxs,new_priorities):
#             self.tree.update(new_priority,idx)

#     def sample(self,batch_size):
#         sampling_probabilities = np.array(self.tree.values)/self.tree.value_sum
#         idxs = np.random.choice(range(self.tree.size),batch_size,replace=False,p=sampling_probabilities)
#         sampling_probabilities = sampling_probabilities[idxs]
#         experiences = [self.buffer[i] for i in idxs]
#         is_weights = np.power(self.tree.size * sampling_probabilities, -self.beta)
#         is_weights /= is_weights.max()
#         is_weights = torch.from_numpy(np.vstack(is_weights)).float().to(device)

#         # increment beta
#         self.beta = min(1.0, self.beta+self.beta_increment_per_sampling)

#         states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
#         actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
#         rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
#         next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
#         dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

#         return states, actions, rewards, next_states, dones, idxs, is_weights
    
#     def __len__(self):
#         return len(self.buffer)