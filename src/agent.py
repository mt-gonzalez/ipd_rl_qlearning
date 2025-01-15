import numpy as np

""" 
    Here it's the implementation of a Q-Learning algorithm to perform the agent to be trained later
    in the Gymnasium environment
"""

class QL_Agent:
    def __init__(self, action_space, observation_space, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.action_space = action_space
        self.observation_space = observation_space
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.q_table = {}  # Initialize Q-table