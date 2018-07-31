import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time


class Actor:
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, action_low, action_high,hidden_size=64):
        """Initialize parameters for the actor.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            action_low (array): Min value of each action dimension
            action_high (array): Max value of each action dimension
        """
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low
        self.num_hidden = hidden_size
        self.build_model()
    
    def inputs(self):
        with tf.name_scope('actor_inputs'):
            inp_state = tf.placeholder(tf.float32,[None,self.state_size],name='state')
            action_gradients = tf.placeholder(tf.float32,[None,self.action_size],name='action_gradients')
            # placeholder for mask used in weighted experience replay
            #replay_buffer_mask = tf.placeholder(tf.float32,[None],name="replay_buffer_mask")
        return inp_state,action_gradients#,replay_buffer_mask
    
    def model(self,inp_state,scope='actor_model'):
        with tf.variable_scope(scope):
            net = slim.fully_connected(inp_state,self.num_hidden,scope='fc1') # (N,num_hidden)
            net = slim.fully_connected(net,self.num_hidden,scope='fc2') # (N,num_hidden)
            net = slim.fully_connected(net,self.action_size,activation_fn=tf.sigmoid,scope='fc3') # (N,action_size)
            net = tf.multiply(net,self.action_range) + self.action_low # map from [0,1] to action ranges
        return net
        
    def loss(self,actions,action_gradients,replay_buffer_mask=None):
        with tf.name_scope('actor_loss'):
            actor_loss = tf.reduce_mean(-actions*action_gradients)
        return actor_loss
             
    def optimizer(self,loss,learning_rate=1e-3):
        with tf.name_scope('actor_optimizer'):
            train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        return train_op
    
    def build_model(self):
        self.inp_state,self.action_gradients = self.inputs()
        self.actions = self.model(self.inp_state)
        self.actor_loss = self.loss(self.actions,self.action_gradients)
        self.opt = self.optimizer(self.actor_loss)