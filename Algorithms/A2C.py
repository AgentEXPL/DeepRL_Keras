import tensorflow as tf
import numpy as np
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko
 
class ACModel(tf.keras.Model):
  def __init__(self, act_dim, is_continuious, act_hidden_size, act_hidden_activation, act_output_activation, v_hidden_size, v_hidden_activation, v_output_activation, **kwargs):
    super().__init__('A2C_net') # name is A2C_net
    self.act_dim = act_dim
    self.is_continuious = is_continuious
    self.act_hidden_size = act_hidden_size
    self.act_hidden_activation = act_hidden_activation
    self.cri_hidden_size = v_hidden_size
    self.v_hidden_activation = v_hidden_activation
    self.act_output_layer = kl.Dense(act_dim, activation = 'act_output_activation')
    self.v_output_layer = kl.Dense(1, activation = 'v_output_activation')
    self.log_std = kwargs.get('log_std') # shall be included if is_continuious == 1
    
  def call(self, inputs): 
    # call function is for defining forward propagation, inputs may be a mini-batch of states/observations, return the outputs
    # actor network representing the policy, the output from actor network is logits if actions are descrete
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
    if self.is_continuious:
     action = tf.squeeze(tf.random.categorical(policy, 1), axis=-1)
     return np.squeeze(action, axis=-1), np.squeeze(v, axis=-1)
    else:
     action = policy_logits + tf.random_normal(tf.shape(policy)) * tf.exp(self.log_std)
     return 
     

def gaussian_likelihood(x, mu, log_std):
 EPS = 1e-8
 pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
 return tf.reduce_sum(pre_sum, axis=1)



class A2CAgent:
 def __init__(self, model):
  self.params = {'value': 0.5, 'entropy': 0.0001}
  self.model = model
  self.model.compile(
   optimizer=ko.RMSprop(lr=0.0007),
   loss=[self._logits_loss, self._value_loss]
  )
 def 
    
    
