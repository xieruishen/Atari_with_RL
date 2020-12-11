# https://medium.com/@ts1829/policy-gradient-reinforcement-learning-in-pytorch-df1383ea0baf
#https://towardsdatascience.com/learning-reinforcement-learning-reinforce-with-pytorch-5e8ad7fc7da0


import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
import multiprocessing
from multiprocessing import Pool
from itertools import starmap





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
              batch_size=5, gamma=0.999):
    # Set up lists to hold results
    total_rewards = []
    batch_rewards = []
    batch_actions = []
    batch_states = []
    batch_counter = 1
    
    # Define optimizer
    optimizer = optim.Adam(policy_estimator.network.parameters(), lr=0.005)
    
    action_space = np.arange(env.action_space.n)
    ep = 0

    

    while ep < num_episodes:
        
        pool_mapping = [(env, policy_estimator, action_space) for i in range(batch_size)]

        batch = starmap(run_episode, pool_mapping)

        for states, actions, rewards in batch:
            
            batch_rewards.extend(discount_rewards(rewards, gamma))

            batch_states.extend(states)
            
            batch_actions.extend(actions)
                           
            total_rewards.append(sum(rewards))
        
                
        #print("\r \n ", "optimizing       ", end="\033[F")

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
            
        avg_rewards = np.mean(total_rewards[-100:])
        # Print running average
        print("\r \n \n", ep, avg_rewards, end="\033[F \033[F")
        ep += batch_size
                
    return total_rewards







def run_episode(env, policy_estimator, action_space):
    print(policy_estimator)
    s_0 = env.reset()
    states = []
    rewards = []
    actions = []
    done = False
    while done == False:
        # Get actions and sample an action
        action_probs = policy_estimator.predict(s_0).detach().numpy()
        action = np.random.choice(action_space, p=action_probs)
        
        #print("\r \n ",len(states), "        step",  end="\033[F")

        # take a step in the environment
        s_1, r, done, _ = env.step(action)

        #print("\r \n ", len(states), "        post step", end="\033[F")
        #append items to our lists
        states.append(s_0)
        rewards.append(r)
        actions.append(action)
        s_0 = s_1

    return (states, actions, rewards)




def reinforce_multiprocess(envs, policy_estimator, num_episodes=2000, gamma=0.999):
    # Set up lists to hold results
    total_rewards = []
    batch_rewards = []
    batch_actions = []
    batch_states = []

    action_space = np.arange(envs[0].action_space.n)

    # Define optimizer
    optimizer = optim.Adam(policy_estimator.network.parameters(), lr=0.005)

    action_space = np.arange(envs[0].action_space.n)
    ep = 0

    while ep < num_episodes:
        
        pool_mapping = [(env, policy_estimator, action_space) for env in envs]

        with Pool(4) as p:
            batch = p.starmap(run_episode, pool_mapping) #p.starmap breaks the model????????

            for states, actions, rewards in batch:
                
                batch_rewards.extend(discount_rewards(rewards, gamma))

                batch_states.extend(states)
                
                batch_actions.extend(actions)
                               
                total_rewards.append(sum(rewards))

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
            
        avg_rewards = np.mean(total_rewards[-100:])
        # Print running average
        print("\r \n \n", ep, avg_rewards, end="\033[F \033[F")
        ep += len(envs)
                
    return total_rewards


if __name__ == "__main__":
    print()
    gym.envs.register(
         id='Cappedlunar-v0',
         entry_point='gym.envs.box2d:LunarLander',
         max_episode_steps=400,
    )

    envs = [gym.make('Cappedlunar-v0')] * 10
    policy_est = policy_estimator_network(envs[0])
    #print(reinforce_multiprocess(envs, policy_est, num_episodes = 1000))
