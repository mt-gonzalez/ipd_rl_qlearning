#Strategies to train the Q-learnig agent in the Iterated Prisoner's Dilemma

""" 
#   "agent_last_action" is the variable corresponding to the last decision made by the Q-learning agent
#   could be:   0 = the agent cooperate
#               1 = the agent defects
# 
#   R = reward, for cooperating both
#   T = temptation, for defect when the opponent cooperate
#   S = sucker, for cooperate when the opponent defect
#   P = penalty, when both defect
#   To make a good Iterated Prisoner's Dilemma, the rewards must follow an order relation: T > R > P > S
"""

import random

def always_cooperate(agent_last_action):
    return 0

def always_defect(agent_last_action):
    return 1

def tft(agent_last_action): #TIT FOR TAT
    """ Cooperates on the first round and imitates its opponent's previous move thereafter """
    return 0 if agent_last_action is None else agent_last_action

def stft(agent_last_action): #SUSPICIOUS TIT FOR TAT
    """ Defects on the first round and imitates its opponent's previous move thereafter """
    return 1 if agent_last_action is None else agent_last_action

def gtft(agent_last_action, R=3, T=5, S=0, P=1): #GENEROUS TIT FOR TAT
    """ Cooprates on the first round and after its opponent cooperates. Following a defection,
        it cooperates with probability """

    coeff_1 = (T - R) / (R - S)
    coeff_2 = (R - P) / (T - P)

    if agent_last_action is None:
        return 0
    
    probabilidad_cooperacion = min(1.0 - coeff_1, coeff_2)

    if agent_last_action == 1:
        return 0 if random.random() < probabilidad_cooperacion else 1
    else:
        return 0

def imptft(agent_last_action): #IMPERFECT TIT FOR TAT
    """ Imitates opponent's last move with high (but less than one) probability """
    if agent_last_action is None:
        return 0
    
    values = [agent_last_action, 1 - agent_last_action]
    probs = [0.8, 0.2]

    selected_response = random.choices(values, probs)[0]

    return selected_response