import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time


class Actor:
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, action_low, action_high,single_rotor_control = False,hidden_size=64,
                    is_training=True):
        """Initialize parameters for the actor.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            action_low (array): Min value of each action dimension
            action_high (array): Max value of each action dimension
        """
        self.state_size = state_size
        self.single_rotor_control = single_rotor_control
        
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low
        self.num_hidden = hidden_size
        self.is_training = is_training
        self.build_model()
    
    def inputs(self):
        with tf.name_scope('actor_inputs'):
            inp_state = tf.placeholder(tf.float32,[None,self.state_size],name='state')
            action_gradients = tf.placeholder(tf.float32,[None,self.action_size],name='action_gradients')
            #is_training = tf.placeholder(tf.bool,name="is_training")
            # placeholder for mask used in weighted experience replay
            #replay_buffer_mask = tf.placeholder(tf.float32,[None],name="replay_buffer_mask")
        return inp_state,action_gradients#,is_training#,replay_buffer_mask
    
    def model(self,inp_state,scope='actor_model'):
        with tf.variable_scope(scope):
            batch_norm_params = {'is_training': self.is_training}
            net = slim.fully_connected(inp_state,self.num_hidden,normalizer_fn=slim.batch_norm,normalizer_params = batch_norm_params,
                                weights_regularizer=slim.l2_regularizer(1e-3),scope='fc1') # (N,num_hidden)
            #net = slim.batch_norm(net,is_training=self.is_training)
            net = slim.fully_connected(net,self.num_hidden,#normalizer_fn=slim.batch_norm,normalizer_params = batch_norm_params,
                                weights_regularizer=slim.l2_regularizer(1e-3),scope='fc2') # (N,num_hidden)

            net = slim.fully_connected(net,self.num_hidden,normalizer_fn=slim.batch_norm,normalizer_params = batch_norm_params,
                               weights_regularizer=slim.l2_regularizer(1e-3),scope='fc3') # (N,num_hidden)
            #net = slim.batch_norm(net,is_training=self.is_training)
            if self.single_rotor_control:
                net = slim.fully_connected(net,1,activation_fn=tf.sigmoid,scope='fc4') # (N,1)
                mask = tf.ones([1,self.action_size])
                net = tf.multiply(net,mask) #(N,action_size)
            else:
                net = slim.fully_connected(net,self.action_size,activation_fn=tf.sigmoid,scope='fc4') # (N,action_size)
            net = tf.multiply(net,self.action_range) + self.action_low # map from [0,1] to action ranges
        return net
        
    def loss(self,actions,action_gradients,replay_buffer_mask=None):
        with tf.name_scope('actor_loss'):
            actor_loss = tf.reduce_mean(-actions*action_gradients)
        return actor_loss
             
    def optimizer(self,loss,learning_rate=1e-4):
        with tf.name_scope('actor_optimizer'):
            #train_op = tf.train.AdamOptimizer(learning_rate)#.minimize(loss)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            gradients, variables = zip(*optimizer.compute_gradients(loss))
            gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
            train_op = optimizer.apply_gradients(zip(gradients, variables))
        return train_op
    
    def build_model(self):
        self.inp_state,self.action_gradients = self.inputs()
        self.actions = self.model(self.inp_state)
        self.actor_loss = self.loss(self.actions,self.action_gradients)
        self.opt = self.optimizer(self.actor_loss)