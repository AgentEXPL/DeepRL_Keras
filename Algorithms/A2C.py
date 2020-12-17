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
    # actor network represents the policy, the output from actor network is logits if actions are descrete
    x = tf.convert_to_tensor(inputs, dtype=tf.float32)
    for i in act_hidden_size:
      x = kl.Dense(i, activation = self.act_hidden_activation)(x)
    self.policy = self.act_output_layer(x)
    
    # critic network reprents the state value, the output from critic network is the value of the state
    y = tf.convert_to_tensor(inputs, dtype=tf.float32)
    for i in v_hidden_size:
      y = kl.Dense(i, activation = self.v_hidden_activation)(y)
    self.v = self.v_output_layer(y)
    return self.policy, self.v
  
  def get_nextAction_v(self, obs):
    policy, v = self.predict(obs)
    if self.is_continuious:
     action = policy + tf.random_normal(tf.shape(policy)) * tf.exp(self.log_std)
     return action, np.squeeze(v, axis=-1)
    else:
     action = tf.squeeze(tf.random.categorical(policy, 1), axis=-1)
     return np.squeeze(action, axis=-1), np.squeeze(v, axis=-1)
     

def gaussian_likelihood(x, mu, log_std):
 EPS = 1e-8
 pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
 return tf.reduce_sum(pre_sum, axis=1)



class A2CAgent:
 def __init__(self, model):
  self.params = {'value': 0.5, 'entropy': 0.0001, 'gamma': 0.99}
  self.model = model
  self.model.compile(
   optimizer=ko.RMSprop(lr=0.0007),
   loss=[self._act_loss, self._cri_loss]
  )
  
 #自定义loss函数，第一个参数是真实值，第二个参数是model根据inputs推导出来的值
 def _cri_loss(self, returns, v):
  return self.params['value']*kls.mean_squared_error(returns, v)
 
 def _act_loss(self, acts_and_advs, policy):
  actions, advantages = tf.split(acts_and_advs, 2, axis=-1)
  actions = tf.cast(actions, tf.int32)
  if self.model.is_continuious: # policy is the logits
   logp = gaussian_likelihood(actions, policy, self.model.log_std)
   return -tf.reduce_mean(logp*advantages) # may be needed double check
  else 
   cross_entropy = kls.CategoricalCrossentropy(from_logits=True)
   policy_loss = cross_entroy(actions, policy, sample_weigth=advantages)
   entropy_loss = cross_entroy(policy, policy)   
   return policy_loss - self.params['entropy']*entropy_loss
 
 def _rets_advs(self, rewards, dones, values, v_next):
  returns = np.append(np.zeros_like(rewards), v_next, axis=-1) # the length of returns = len(rewards) +1
  for t in range(rewards.shape[0])[::-1]:
   returns[t] = rewards[t] + self.params['gamma']*returns[t+1]*(1-dones[t])
  returns = returns[:-1] # len(returns) -= 1
  advantages = returns - values
  return returns, advantages
 
 def train(self, env, batch_sz=32, updates=1000): # need double check, especially about the dim of variables related to actions
  # storage helpers for a single batch of data
  actions = np.empty((batch_sz,), dtype=np.int32) # act_dim may be impacted by is_continuious
  rewards, dones, values = np.empty((3, batch_sz))
  observations = np.empty((batch_sz,) + env.observation_space.shape) # env.observation_space.shape may be impacted by is_continuious
  # training loop: collect samples, send to optimizer, repeat updates times
  ep_rews = [0.0]
  next_obs = env.reset()
  for update in range(updates):
   for step in range(batch_sz):
    observations[step] = next_obs.copy()
    actions[step], values[step] = self.model.get_nextAction_v(next_obs[None, :]) # None means what?
    next_obs, rewards[step], dones[step], _ = env.step(actions[step])
    ep_rews[-1] += rewards[step]
    if dones[step]:
     ep_rews.append(0.0)
     next_obs = env.reset()
   _, next_value = self.model.get_nextAction_v(next_obs[None, :])
   returns, advs = self._rets_advs(rewards, dones, values, next_value)
   # a trick to input actions and advantages through same API
   acts_and_advs = np.concatenate([actions[:, None], advs[:, None]], axis=-1)
   # performs a full training step on the collected batch
   # note: no need to mess around with gradients, Keras API handles it
   # actor和critic网络同时更新，需要改进，to be continued
   losses = self.model.train_on_batch(observations, [acts_and_advs, returns])
  return ep_rews

 def test(self, env, render=True):
  obs, done, ep_reward = env.reset(), False, 0
  while not done:
   action, _ = self.model.action_value(obs[None, :])
   obs, reward, done, _ = env.step(action)
   ep_reward += reward
   if render:
    env.render()
   return ep_reward
 

import gym
env = gym.make('CartPole-v0')
model = ACModel(env.action_space.n, 0, [128], 'relu', None, [128], 'relu', None)
obs = env.reset()
# no feed_dict or tf.Session() needed at all
action, value = model.get_nextAction_v(obs[None, :])
print(action, value) # [1] [-0.00145713]
agent = A2CAgent(model)
rewards_sum = agent.test(env)
print("%d out of 200" % rewards_sum) # 18 out of 200
rewards_history = agent.train(env)
print("Finished training, testing...")
print("%d out of 200" % agent.test(env)) # 200 out of 200
