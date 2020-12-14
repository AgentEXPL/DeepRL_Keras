import tensorflow as tf
import numpy as np
import tensorflow.keras.layers as kl
 
class Model(tf.keras.Model):
  def __init__(self, act_dim, is_continuious, act_hidden_size, act_hidden_activation, act_output_activation, v_hidden_size, v_hidden_activation, v_output_activation):
    super().__init__('A2C_net') # name is A2C_net
    self.act_dim = act_dim
    self.is_continuious = is_continuious
    self.act_hidden_size = act_hidden_size
    self.act_hidden_activation = act_hidden_activation
    self.cri_hidden_size = v_hidden_size
    self.v_hidden_activation = v_hidden_activation
    self.act_output_layer = kl.Dense(act_dim, activation = 'act_output_activation')
    self.v_output_layer = kl.Dense(1, activation = 'v_output_activation')
    
  def call(self, inputs): 
    # call function is for defining forward propagation, inputs may be a mini-batch of states/observations, return the outputs
    # actor network representing the policy
    x = tf.convert_to_tensor(inputs, dtype=tf.float32)
    for i in act_hidden_size:
      x = kl.Dense(i, activation = self.act_hidden_activation)(x)
    self.policy = self.act_output_layer(x)
    
    # critic network reprenting the state value
    y = tf.convert_to_tensor(inputs, dtype=tf.float32)
    for i in v_hidden_size:
      y = kl.Dense(i, activation = self.v_hidden_activation)(y)
    self.v = self.v_output_layer(y)
    return self.policy, self.v
  
  def get_action(self, obs):
    policy, v = self.predict(obs)
    action 
  
