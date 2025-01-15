import sys
import os

#Adding src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest
import numpy as np
from agent import QL_Agent

@pytest.fixture
def agent():
    #Initializing a test agent
    return QL_Agent(action_space=2, observation_space=4)

def test_initial_q_table(agent):
    #Check that the Q-table is initialized with zeros
    assert np.array_equal(agent.q_table, np.zeros((4, 2)))

def test_get_state_key(agent):
    #State mapping test
    observation = (0, 0)  # State: (Cooperate, Cooperate)
    state_key = agent._get_state_key(observation)
    assert state_key == 0
    observation = (1, 0)  # State: (Defect, Cooperate)
    state_key = agent._get_state_key(observation)
    assert state_key == 2

def test_choose_action_exploitation(agent):
    #Manipulating agent for exploitation
    agent.epsilon = 0
    #Modified Q-table for state 0
    agent.q_table = np.array([[0, 1], [0, 0], [0, 0], [0, 0]])  # Action 1 has higher Q-value for state 0
    
    observation = (0, 0)  # State: (Cooperate, Cooperate)
    action = agent.choose_action(observation)
    
    #Agent should choose action 1 (higher Q-value)
    assert action == 1

def test_choose_action_exploration(agent):
    #Exploration test (epsilon = 1, es decir, agent choose random action)
    agent.epsilon = 1
    observation = (0, 0)
    action = agent.choose_action(observation)
    #Because it's a random action, I just check if it's a valid value
    assert action in [0, 1]

def test_td_target_func(agent):
    #Checking TD function
    agent.q_table = np.array([[0, 0], [0, 1], [1, 0], [0, 0]])
    reward = 1
    next_state_index = 1  #Next state
    td_target = agent.td_target_func(reward, next_state_index)
    
    #Formula = R + gamma * max_a' Q(s', a'), so the result must be:
    #1 + 0.9 * 1 = 1.9
    assert np.isclose(td_target, 1.9)

def test_q_value_func(agent):
    #Initial Q-table configuration
    agent.q_table = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    state_index = 0
    action = 0
    td_target = 1.5
    initial_q_value = agent.q_table[state_index, action]
    
    #Updating Q-value for pair (state, action)
    agent.q_value_func(state_index, action, td_target)
    
    #Calculating expected value
    expected_value = initial_q_value + agent.alpha * (td_target - initial_q_value)
    updated_q_value = agent.q_table[state_index, action]
    
    #Checking if updated value match expected value
    assert np.isclose(updated_q_value, expected_value)

def test_learn(agent):
    #Is the agent learning correctly? ->
    initial_q_value = agent.q_table[0, 0]
    observation = (0, 0)
    action = 0
    reward = 1
    next_observation = (1, 0)
    done = False
    
    #Learning func. call
    agent.learn(observation, action, reward, next_observation, done)
    
    #Q-table updated correctly ->
    updated_q_value = agent.q_table[0, 0]
    assert updated_q_value != initial_q_value  #Q-value should be updated