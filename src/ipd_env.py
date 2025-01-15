import gymnasium as gym
from gymnasium import spaces
import numpy as np

class IteratedPrisonersDilemmaEnv(gym.Env):
    #It will be a default opponent strategy if a list of them isn't submitted
    def __init__(self, num_rounds=10, R=3, T=5, S=0, P=1, opponent_strategies=None):
        
        #Binary action space: Coop -> 0, Betray -> 1
        self.action_space = spaces.Discrete(2)

        #MultiDiscrete space: Array with Agent and Opponent decision
        self.observation_space = spaces.MultiDiscrete([2, 2])

        #Environment configuration
        self.num_rounds = num_rounds
        self.R, self.T, self.S, self.P = R, T, S, P #Reward, Temptation, Sucker, Penalty
        self.current_round = 0
        self.agent_history = []  #Agent decision history
        self.opponent_history = []  #Opponent decision history

        # Default strategy
        self.default_strategy = self.tit_for_tat

        if opponent_strategies is None:
            self.opponent_strategies = [self.default_strategy]
        else:
            self.opponent_strategies = opponent_strategies
        
        self.current_opponent = None  # Actual opponent strategy

        #Reward matrix (coop, betray)
        self.reward_matrix = {
            (0, 0): (self.R, self.R),  # Both coop
            (0, 1): (self.S, self.T),  # Agent coop, opponent defects
            (1, 0): (self.T, self.S),  # Agent defects, opponent coop
            (1, 1): (self.P, self.P)   # Both defect
        }

    def tit_for_tat(self, agent_last_action):
        """Copies last Agent decision. Coop if decide first."""
        return 0 if agent_last_action is None else agent_last_action

    def _get_observation(self):
        # Last decisions (compact state)
        last_agent_action = self.agent_history[-1] if self.agent_history else 0
        last_opponent_action = self.opponent_history[-1] if self.opponent_history else 0
        last_decision = [last_agent_action, last_opponent_action]
        
        return last_decision
    
    def reset(self):
        # Reset the game state
        self.current_round = 0
        self.agent_history = []  # Clear agent's decision history
        self.opponent_history = []  # Clear opponent's decision history

        # Select a random opponent strategy for the new episode
        self.current_opponent = np.random.choice(self.opponent_strategies)

        # Return the initial observation
        return self._get_observation()
    
    def step(self, action):
    # Ensure action is either 0 or 1 (coop or defect)
        if action not in [0, 1]:
            raise ValueError("Action must be 0 (Cooperate) or 1 (Defect)")

        # If this is the first round, let the opponent make the first decision
        if self.current_round == 0:
            # Opponent makes the first move based on its strategy
            opponent_action = self.current_opponent(None)  # No previous action at the start
        else:
            # In subsequent rounds, the opponent's decision is based on the agent's previous move
            opponent_action = self.current_opponent(self.agent_history[-1])

        # Append the agent's action and opponent's action to the history
        self.agent_history.append(action)
        self.opponent_history.append(opponent_action)

        # Calculate the reward based on the current round's decisions
        agent_reward, opponent_reward = self.reward_matrix[(action, opponent_action)]

        # Increment the round
        self.current_round += 1

        # Check if the game has ended
        done = self.current_round >= self.num_rounds

        # Get the observation after the action
        observation = self._get_observation()

        # Return observation, reward, done flag, and any additional info
        return observation, agent_reward, done, {}

    def render(self):
        """
        Print current game history.
        """
        print(f"Round: {self.current_round}")
        print(f"Agent History: {self.agent_history}")
        print(f"Opponent History: {self.opponent_history}")