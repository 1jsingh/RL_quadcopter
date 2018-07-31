from task import Task

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time

from agents.actor import Actor
from agents.critic import Critic
from agents.utils import *

class DDPG():
    """Reinforcement Learning agent that learns using DDPG."""
    def __init__(self, task):
        tf.reset_default_graph()
        
        self.task = task
        self.state_size = self.task.state_size
        self.action_size = self.task.action_size
        self.action_low = self.task.action_low
        self.action_high = self.task.action_high
        
        with tf.variable_scope("local"):
            self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
            self.critic_local = Critic(self.state_size, self.action_size)
        
        with tf.variable_scope("target"):
            self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
            self.critic_target = Critic(self.state_size, self.action_size)
            
        
        # Initialize target model parameters with local model parameters
        #self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        #self.actor_target.model.set_weights(self.actor_local.model.get_weights())
        
        self.tau = tf.placeholder(tf.float32,name='tau')
        self.target_update_ops = self.soft_update()
        
        with tf.name_scope('summary'):
            self.reward_log = tf.Variable(0.,False,name='reward_log',dtype=tf.float32)
            self.eps_length_log = tf.Variable(0.,False,name='reward_log',dtype=tf.float32)
            tf.summary.scalar('reward_log', self.reward_log)
            tf.summary.scalar('eps_length_log', self.eps_length_log)
            self.summary_op = tf.summary.merge_all()
            
        # Noise process
        self.exploration_mu = 0
        self.exploration_theta = (self.action_high - self.action_low)*.2
        self.exploration_sigma = (self.action_high - self.action_low)*.2
        self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)

        # Replay memory
        self.buffer_size = 100000
        self.batch_size = 64
        self.memory = ReplayBuffer(self.buffer_size,self.batch_size)

        # Algorithm parameters
        self.gamma = 0.99  # discount factor

    def reset(self):
        self.noise.reset()
        state = self.task.reset()
        self.last_state = state
        return state

    def step(self, sess,action, reward, next_state, done):
        # Save experience / reward
        self.memory.add(self.last_state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(sess,experiences)

        # Roll over last state and action
        self.last_state = next_state

    def act(self,sess,state):
        """Returns actions for given state(s) as per current policy."""
        state = np.reshape(state, [-1, self.state_size])
        action = sess.run(self.actor_local.actions,feed_dict={self.actor_local.inp_state:state})
        #return list(action + self.noise.sample())  # add some noise for exploration
        return np.clip((action[0] + self.noise.sample()),self.action_low,self.action_high)
        
    def learn(self, sess,experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        #for e in experiences:
        #    print (e.state.shape,e.done)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Get predicted next-state actions and Q values from target models
        #     Q_targets_next = critic_target(next_state, actor_target(next_state))
        actions_next = sess.run(self.actor_local.actions,feed_dict={self.actor_local.inp_state:next_states})
        Q_values_next = sess.run(self.critic_local.q_pred,feed_dict={self.critic_local.inp_state:next_states
                                                                     ,self.critic_local.actions:actions_next})

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_values_next * (1 - dones)
        sess.run(self.critic_target.opt,feed_dict={self.critic_target.inp_state:states,
                                                   self.critic_target.actions:actions,
                                                  self.critic_target.qtarget:Q_targets})

        # Train actor model (local)
        ag = sess.run(self.critic_local.action_gradients,feed_dict={self.critic_local.inp_state:states,
                                                                    self.critic_local.actions:actions})
        action_gradients = np.reshape(ag, (-1, self.action_size))
        sess.run(self.actor_target.opt,feed_dict={self.actor_target.inp_state:states,
                                                  self.actor_target.action_gradients:action_gradients})
        # Soft-update target models
        sess.run(self.target_update_ops,feed_dict={self.tau:0.1})   

    def soft_update(self):
        local_list = slim.get_variables_to_restore(include=["local"])
        target_list = slim.get_variables_to_restore(include=["target"])
        update_ops = []
        for i in range(len(local_list)):
            update_op = tf.assign(local_list[i],(1-self.tau)*local_list[i] + (self.tau)*target_list[i])
            update_ops.append(update_op)
        return update_ops
    
    def train(self,max_num_episodes=10000):
        
        # getting timestamp
        timestamp = str(int(time.time()))

        # log_path for tensorboard
        LOGS_PATH = "./tensorboard/"+timestamp
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            # create log writer object
            train_writer = tf.summary.FileWriter(LOGS_PATH+'/train',sess.graph)
            
            # Set same weights for target and local models
            sess.run(self.target_update_ops,feed_dict={self.tau:1.0})
            DISPLAY_STEP = 100
            max_reward = -10000
            for eps_count in range(max_num_episodes):
                self.reset()
                total_reward,episode_step = 0,0
                while True:
                    action = self.act(sess,self.last_state)
                    next_state,reward,done = self.task.step(action)
                    #print (step,self.last_state,action,next_state)
                    self.step(sess,action, reward, next_state, done)
                    total_reward += reward
                    episode_step += 1
                    
                    if done or episode_step == 200:
                        max_reward = max(max_reward,total_reward)
                        if eps_count % DISPLAY_STEP == 0:
                            print ("episode : {} ... total reward :{} ... max reward :{} ... eps_length : {}".format(eps_count,
                                                                                                 total_reward,max_reward,episode_step))
                        summary = sess.run(self.summary_op,feed_dict={self.reward_log:total_reward,self.eps_length_log:episode_step})
                        train_writer.add_summary(summary, eps_count)
                        break
