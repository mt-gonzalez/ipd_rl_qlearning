import sys
import os

#Adding src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest
import numpy as np
from ipd_env import IteratedPrisonersDilemmaEnv

def test_initialization():
    env = IteratedPrisonersDilemmaEnv()
    
    assert env.num_rounds == 10
    assert env.R == 3 and env.T == 5 and env.S == 0 and env.P == 1
    assert env.current_round == 0
    assert env.agent_history == []
    assert env.opponent_history == []
    assert env.current_opponent is None
    assert len(env.opponent_strategies) == 1
    assert env.opponent_strategies[0] == env.tit_for_tat

def test_reset():
    env = IteratedPrisonersDilemmaEnv()
    observation = env.reset()

    assert env.current_round == 0
    assert env.agent_history == []
    assert env.opponent_history == []
    assert env.current_opponent in env.opponent_strategies
    assert observation == [0, 0]  #Initial default observation

def test_get_observation():
    env = IteratedPrisonersDilemmaEnv()
    env.agent_history = [0, 1]
    env.opponent_history = [1, 0]
    observation = env._get_observation()

    assert observation == [1, 0]

def test_step():
    env = IteratedPrisonersDilemmaEnv(num_rounds=5)
    env.reset()
    env.current_opponent = env.tit_for_tat  #Fix opponent strategy
    
    observation, reward, done, _ = env.step(0)  #Agent cooperates in the first round
    
    # Verificar actualizaciones
    assert observation == [0, 0]  # Tit-for-tat cooperate in the first round
    assert reward == env.R  #Both cooperate
    assert not done  #Game isn't done
    assert env.agent_history == [0]
    assert env.opponent_history == [0]

    #Second round: Agent defects
    observation, reward, done, _ = env.step(1)
    assert observation == [1, 0]  #Agent defects, opponent cooperate
    assert reward == env.T  #Temptation reward
    assert not done  #Game isn't done
    assert env.agent_history == [0, 1]
    assert env.opponent_history == [0, 0]

    #Iterate until the game end
    for _ in range(3):
        observation, reward, done, _ = env.step(0)
    assert done  #Game has ended

def test_tit_for_tat():
    env = IteratedPrisonersDilemmaEnv()
    strategy = env.tit_for_tat

    #First round: no previous decision
    assert strategy(None) == 0  #Start cooperating by default

    #Second round: copies opponent decision
    assert strategy(0) == 0  #Cooperate
    assert strategy(1) == 1  #Defects

def test_reward_matrix():
    env = IteratedPrisonersDilemmaEnv()

    assert env.reward_matrix[(0, 0)] == (env.R, env.R)  #Both cooperate
    assert env.reward_matrix[(0, 1)] == (env.S, env.T)  #Agent cooperate, opponent defects
    assert env.reward_matrix[(1, 0)] == (env.T, env.S)  #Agent defects, opponent cooperate
    assert env.reward_matrix[(1, 1)] == (env.P, env.P)  #Both defect