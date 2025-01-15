import numpy as np
import random

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

        #Initialize Q-table as a numpy array: 4 states (last decision combinations) x 2 actions
        self.q_table = np.zeros((4, 2))  # 4 states, 2 possible actions (Cooperate, Defect)

    def _get_state_key(self, observation):
        """
        Convert observation to a state key.
        In this case, the state is represented by the pair of last decisions of the agent and opponent.
        """
        last_agent_action, last_opponent_action = observation
        
        # Map the state (last agent action, last opponent action) to an index in the Q-table
        # The states are: (Cooperate, Cooperate), (Cooperate, Defect), (Defect, Cooperate), (Defect, Defect)
        state_map = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}
        
        # Return the corresponding state index (0-3)
        return state_map[(last_agent_action, last_opponent_action)]

    def choose_action(self, observation):
        """
        Choose an action using epsilon-greedy policy.
        """
        state_index = self._get_state_key(observation)

        # Epsilon-greedy action selection
        if random.uniform(0, 1) < self.epsilon:
            # Exploration: Choose random action
            action = np.random.choice([0, 1])  # Random action
        else:
            # Exploitation: Choose the best known action (the one with the highest Q-value)
            action = np.argmax(self.q_table[state_index])

        return action

    def td_target_func(self, reward, next_state_index):
        """
        Compute the TD target: R + gamma * max_a' Q(s', a')
        """
        # Find the best future Q-value for the next state
        best_next_action = np.argmax(self.q_table[next_state_index])
        return reward + self.gamma * self.q_table[next_state_index, best_next_action]

    def q_value_func(self, state_index, action, td_target):
        """
        Update the Q-value for a given state-action pair using the TD target.
        """
        # Calculate TD error
        td_error = td_target - self.q_table[state_index, action]
        # Update the Q-value
        self.q_table[state_index, action] += self.alpha * td_error

    def learn(self, observation, action, reward, next_observation, done):
        """
        Update Q-table based on the agent's experience.
        """
        # Get state indices for the current and next observations
        state_index = self._get_state_key(observation)
        next_state_index = self._get_state_key(next_observation)

        # Compute the TD target
        td_target = self.td_target_func(reward, next_state_index)

        # Update the Q-value
        self.q_value_func(state_index, action, td_target)