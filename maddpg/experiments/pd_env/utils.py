import numpy as np
import pandas as pd
import pd_env.CONST as CONST
from typing import List, Tuple
from copy import copy

def pd_pairs(agents: List[int]) -> List[Tuple[int]]:
    """
    Randomly pairs agents into two groups for the prisoner's dilemma game.

    Args:
        agents (List[int]): A list of agent IDs.

    Returns:
        List[Tuple[int]]: A list containing two tuples, where the first tuple represents the first pair of agents
                          and the second tuple represents the second pair of agents.
    """
    # assert len(agents) == CONST.NUM_AGENTS, f"Number of agents must be {CONST.NUM_AGENTS}."
    first_pair = np.random.choice(agents, 2, replace=False)
    second_pair = set(agents) - set(first_pair)
    return [tuple(first_pair), tuple(second_pair)]

from typing import List, Tuple

def my_partner(agent: int, pairs: Tuple[List[int]]) -> int:
    """
    Find the partner of the given agent in a list of pairs.

    Args:
        agent (int): The agent for which to find the partner.
        pairs (Tuple[List[int]]): A list of pairs, where each pair is represented as a list of two agents.

    Returns:
        int: The partner of the given agent.

    Raises:
        None

    Examples:
        >>> my_partner(1, [(1, 2), (3, 4)])
        2
        >>> my_partner(3, [(1, 2), (3, 4)])
        4
    """
    for pair in pairs:
        if agent in pair:
            partner = set(pair) - {agent}
            if len(partner) == 1:
                return int(list(partner)[0])
            return list(partner)

def pretty_print_obs(observations: List[Tuple]) -> pd.DataFrame:
    """
    Converts a list of observations into a pandas DataFrame and returns it.

    Args:
        observations (List[Tuple]): A list of tuples representing observations. Each tuple should contain the following elements:
            - agent: The agent's identifier.
            - pairs: A list of tuples representing pairs of agents.
            - actions: A dictionary mapping agents to their chosen actions.
            - payoffs: A dictionary mapping agents to their received payoffs.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the following columns:
            - Round: The round number.
            - Agent: The agent's identifier.
            - Partner: The partner's identifier.
            - Actions: The chosen actions of the agent and its partner.
            - Rewards: The received payoffs of the agent and its partner.
    """

    lst = []
    for i, obs in enumerate(observations):
        agent, pairs, actions, payoffs = obs[0], obs[2], obs[3], obs[4]
        partner = my_partner(agent, pairs)
        try:
            my_action = CONST.ACTIONS_NAMES[actions[agent]] 
        except:
            my_action = "None"
        try:
            partner_action = CONST.ACTIONS_NAMES[actions[partner]]
        except:
            partner_action = "None"
        my_payoff, partner_payoff = payoffs[agent], payoffs[partner]
        
        dct = {
            'Round': i+1,
            'Agent': agent,
            'Partner': partner,
            'Actions': (my_action, partner_action),
            'Rewards': (my_payoff, partner_payoff)
        }
        lst.append(dct)

    return pd.DataFrame(lst).set_index('Round')