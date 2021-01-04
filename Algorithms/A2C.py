import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko
import gym
import logging
from gym import wrappers



class ACModel(tf.keras.Model):
    def __init__(self, act_dim, is_continuious, act_hidden_size, act_hidden_activation, act_output_activation, v_hidden_size, v_hidden_activation, v_output_activation, **kwargs):
        super().__init__('A2C_net') # name is A2C_net
        self.act_dim = act_dim
        self.is_continuious = is_continuious # This equals to 0, if actions are descrete!!! In this file, only descrete actions are currently support.
        # the layers of the network model is defined in init function
        self.act_hidden_layer = kl.Dense(act_hidden_size, activation = act_hidden_activation)
        self.v_hidden_layer = kl.Dense(v_hidden_size, activation = v_hidden_activation)
        self.act_output_layer = kl.Dense(act_dim, activation = act_output_activation)
        self.v_output_layer = kl.Dense(1, activation = v_output_activation)

    
    def call(self, inputs): 
    # call function is for defining forward propagation, inputs may be a mini-batch of states/observations, return the outputs
    # actor network represents the policy, the output from actor network is logits if actions are descrete
        x = tf.convert_to_tensor(inputs, dtype=tf.float32)
        x = self.act_hidden_layer(x)
        self.policy = self.act_output_layer(x)
    
    # critic network reprents the state value, the output from critic network is the value of the state
        y = tf.convert_to_tensor(inputs, dtype=tf.float32)
        y = self.v_hidden_layer(y)
        self.v = self.v_output_layer(y)
        return self.policy, self.v
  
    def get_nextAction_v(self, obs):
        policy, v = self.predict(obs)
        action = tf.squeeze(tf.random.categorical(policy, 1), axis=-1)
        return np.squeeze(action, axis=-1), np.squeeze(v, axis=-1)
     

class A2CAgent:
    def __init__(self, model):
        self.params = {'value': 0.5, 'entropy': 0.0001, 'gamma': 0.99}
        self.model = model
        self.optimizer=ko.RMSprop(lr=0.007)

    def action_loss(self, actions, advantages, policy_prediction):
        actions = tf.cast(actions, tf.int32)
        policy_loss = kls.SparseCategoricalCrossentropy(from_logits=True)(actions, policy_prediction, sample_weight=advantages)
        policy_2 = tf.nn.softmax(policy_prediction)
        entropy_loss = kls.categorical_crossentropy(policy_2, policy_2)
        return policy_loss - self.params['entropy']*entropy_loss 

    def value_loss(self, returns, v_prediction):
        returns = tf.convert_to_tensor(returns)
        return self.params['value']*kls.mean_squared_error(returns, v_prediction)
 
    def _rets_advs(self, rewards, dones, values, v_next):
        returns = np.append(np.zeros_like(rewards), v_next, axis=-1) # the length of returns = len(rewards) +1
        for t in range(rewards.shape[0])[::-1]:
            returns[t] = rewards[t] + self.params['gamma']*returns[t+1]*(1-dones[t])
        returns = returns[:-1] # len(returns) -= 1
        advantages = returns - values
        return returns, advantages
 
    def train(self, env, batch_sz=64, updates=250): 
        actions = np.empty((batch_sz,), dtype=np.int32)
        rewards, dones, values = np.empty((3, batch_sz))
        observations = np.empty((batch_sz,) + env.observation_space.shape) # env.observation_space.shape may be impacted by is_continuious
        # training loop: collect samples, send to optimizer, repeat updates times
        ep_rews = [0.0]
        next_obs = env.reset()
        for update in range(updates):
            for step in range(batch_sz):
                observations[step] = next_obs.copy()
                actions[step], values[step] = self.model.get_nextAction_v(next_obs[None, :]) # None means what
                next_obs, rewards[step], dones[step], _ = env.step(actions[step])
                ep_rews[-1] += rewards[step]
                if dones[step]:
                    ep_rews.append(0.0)
                    next_obs = env.reset()
                    logging.info("Episode: %03d, Reward: %03d" % (len(ep_rews) - 1, ep_rews[-2]))
            _, next_value = self.model.get_nextAction_v(next_obs[None, :])
            returns, advs = self._rets_advs(rewards, dones, values, next_value)
            
            # This is an on-policy algorithm, i.e., the samples used for each update are generated based on the network model (e.g., policy) to be updated.
            
            with tf.GradientTape() as tape:
                policy_prediction, v_prediction = self.model(observations)
                action_loss = self.action_loss(actions[:,None], advs[:,None], policy_prediction)
                value_loss = self.value_loss(returns[:,None], v_prediction)
                action_loss = tf.reduce_mean(action_loss)
                value_loss = tf.reduce_mean(value_loss)
            grads = tape.gradient([action_loss,value_loss], self.model.variables)
            self.optimizer.apply_gradients(grads_and_vars=zip(grads, self.model.variables))
            logging.debug("[%d/%d] Action-Losses: %s" % (update + 1, updates, action_loss))
            logging.debug("[%d/%d] Value-Losses: %s" % (update + 1, updates, value_loss))
        return ep_rews
            


    def test(self, env, render=True):
        outdir = '/tmp/A2C-Cartpole-agent-results'
        env = wrappers.Monitor(env, directory=outdir, force=True) # This is for showing the result as a video.
        obs, done, ep_reward = env.reset(), False, 0
        while not done:
            action, _ = self.model.get_nextAction_v(obs[None, :])
            obs, reward, done, _ = env.step(action)
            ep_reward += reward
        return ep_reward

    
    
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)

    env = gym.make('CartPole-v0')
    model = ACModel(env.action_space.n, 0, 128, 'relu', None, 128, 'relu', None)
    obs = env.reset()
    action, value = model.get_nextAction_v(obs[None, :])
    print(action, value) # 
    agent = A2CAgent(model)
    rewards_sum = agent.test(env)
    print("%d out of 200" % rewards_sum) # The score in one test with random agent.
    rewards_history = agent.train(env)
    print("Finished training, testing...")
    print("%d out of 200" % agent.test(env)) # The score in one test with trained agent.
    
    #Plot the rewards of history episodes during tranining
    plt.plot(np.arange(0,len(rewards_history),1), rewards_history[::1])
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show
