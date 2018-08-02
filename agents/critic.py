import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time

class Critic:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size,hidden_size=64,is_training=True):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size
        self.num_hidden = hidden_size
        self.is_training = is_training
        self.build_model()

    def inputs(self):
        with tf.name_scope('critic_inputs'):
            inp_state = tf.placeholder(tf.float32,[None,self.state_size],name='state')
            actions = tf.placeholder(tf.float32,[None,self.action_size],name='actions')
            qtarget = tf.placeholder(tf.float32,[None,1],name='qtarget')
            #replay_buffer_mask = tf.placeholder(tf.float32,[None],name="replay_buffer_mask")
        return inp_state,actions,qtarget#,replay_buffer_mask
    
    def model(self,inp_state,actions,scope='critic_model'):
        with tf.variable_scope(scope):
            batch_norm_params = {'is_training': self.is_training}
            with tf.variable_scope("state"):
                net_states = slim.fully_connected(inp_state,self.num_hidden,weights_regularizer=slim.l2_regularizer(1e-3),
                            normalizer_fn=slim.batch_norm,normalizer_params = batch_norm_params,scope='fc1') # (N,num_hidden)
                net_states = slim.fully_connected(inp_state,self.num_hidden,weights_regularizer=slim.l2_regularizer(1e-3),
                            normalizer_fn=slim.batch_norm,normalizer_params = batch_norm_params,scope='fc2') # (N,num_hidden)
                
            
            with tf.variable_scope("action"):
                net_actions = slim.fully_connected(actions,self.num_hidden,normalizer_fn=slim.batch_norm,weights_regularizer=slim.l2_regularizer(1e-3),
                                            normalizer_params = batch_norm_params,scope='fc1') # (N,num_hidden)
                net_actions = slim.fully_connected(actions,self.num_hidden,normalizer_fn=slim.batch_norm,weights_regularizer=slim.l2_regularizer(1e-3),
                                            normalizer_params = batch_norm_params,scope='fc2') # (N,num_hidden)
            
            with tf.name_scope("combined"):
                net = tf.add(net_states,net_actions)
                net = slim.fully_connected(net,self.num_hidden,normalizer_fn=slim.batch_norm,normalizer_params = batch_norm_params,
                                                weights_regularizer=slim.l2_regularizer(1e-3),scope='fc1_combined') # (N,num_hidden)
                net = slim.fully_connected(net,1,activation_fn=None,scope='net')
            
            with tf.name_scope("action_gradient"):
                action_gradients = tf.gradients(net,actions)
            
        return net,action_gradients
        
    def loss(self,qtarget,q_pred,replay_buffer_mask=None):
        with tf.name_scope('critic_loss'):
            critic_loss = tf.reduce_mean(tf.square(qtarget-q_pred))
        return critic_loss
             
    def optimizer(self,loss,learning_rate=1e-3):
        with tf.name_scope('critic_optimizer'):
            #train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            gradients, variables = zip(*optimizer.compute_gradients(loss))
            gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
            train_op = optimizer.apply_gradients(zip(gradients, variables))
        return train_op
    
    def build_model(self):
        self.inp_state,self.actions,self.qtarget = self.inputs()
        self.q_pred,self.action_gradients = self.model(self.inp_state,self.actions)
        self.critic_loss = self.loss(self.qtarget,self.q_pred)
        self.opt = self.optimizer(self.critic_loss)