# https://medium.com/@ts1829/policy-gradient-reinforcement-learning-in-pytorch-df1383ea0baf
#https://towardsdatascience.com/learning-reinforcement-learning-reinforce-with-pytorch-5e8ad7fc7da0


import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical



class policy_estimator_network():
    def __init__(self, env):
        self.n_inputs = env.observation_space.shape[0]
        self.n_outputs = env.action_space.n
        
        # Define network
        self.network = nn.Sequential(
            nn.Linear(self.n_inputs, 16), 
            nn.ReLU(), 
            nn.Linear(16, self.n_outputs),
            nn.Softmax(dim=-1))
    
    def predict(self, state):
        action_probs = self.network(torch.FloatTensor(state))
        return action_probs


def discount_rewards(rewards, gamma=0.99):
    r = np.array([gamma**i * rewards[i] 
        for i in range(len(rewards))])
    # Reverse the array direction for cumsum and then
    # revert back to the original order
    r = r[::-1].cumsum()[::-1]
    return r - r.mean()


def reinforce(env, policy_estimator, num_episodes=2000,
              batch_size=10, gamma=0.99):
    # Set up lists to hold results
    total_rewards = []
    batch_rewards = []
    batch_actions = []
    batch_states = []
    batch_counter = 1
    
    # Define optimizer
    optimizer = optim.Adam(policy_estimator.network.parameters(), lr=0.01)
    
    action_space = np.arange(env.action_space.n)
    ep = 0

    while ep < num_episodes:
        s_0 = env.reset()
        states = []
        rewards = []
        actions = []
        done = False
        while done == False:
            # Get actions and sample an action
            action_probs = policy_estimator.predict(s_0).detach().numpy()
            action = np.random.choice(action_space, p=action_probs)
            
            # take a step in the environment
            s_1, r, done, _ = env.step(action)
            
            #append items to our lists
            states.append(s_0)
            rewards.append(r)
            actions.append(action)
            s_0 = s_1
            
            # If done, batch data
            if done:
                batch_rewards.extend(discount_rewards(rewards, gamma))
                
                batch_states.extend(states)
                
                batch_actions.extend(actions)
                
                batch_counter += 1
               
                total_rewards.append(sum(rewards))
                
                # If batch is complete, update network
                if batch_counter == batch_size:
                    
                    #set gradients to 0
                    optimizer.zero_grad()

                    #create tensors of our states and rewards
                    state_tensor = torch.FloatTensor(batch_states)
                    reward_tensor = torch.FloatTensor(batch_rewards)
                    action_tensor = torch.LongTensor(batch_actions)
                    
                    # Calculate the log probability of all actions at all states
                    logprob = torch.log(policy_estimator.predict(state_tensor))

                    action_tensor = torch.reshape(action_tensor, (len(action_tensor), 1))

                    selected_logprobs = reward_tensor * torch.gather(logprob, 1, action_tensor).squeeze()

                    loss = -selected_logprobs.mean()
                    
                    # Calculate gradients
                    loss.backward()
                    # Apply gradients
                    optimizer.step()
                    
                    batch_rewards = []
                    batch_actions = []
                    batch_states = []
                    batch_counter = 1
                    
                avg_rewards = np.mean(total_rewards[-100:])
                # Print running average
                print("\r", ep, avg_rewards, end="")
                ep += 1
                
    return total_rewards



env = gym.make('LunarLander-v2')
policy_est = policy_estimator_network(env)
rewards = reinforce(env, policy_est, num_episodes = 1000)


s_0 = env.reset()
action_space = np.arange(env.action_space.n)
done = False

for _ in range(1000):
    env.render()
    if not done:
        action_probs = policy_est.predict(s_0).detach().numpy()
        action = np.random.choice(action_space, p=action_probs)
        s_1, r, done, _ = env.step(action)
        s_0 = s_1
    else: break
env.close()