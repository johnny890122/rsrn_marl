import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
from itertools import combinations
import random
from scipy.special import comb
from typing import Dict, Tuple, List
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
import CONST 
import functools
import utils

class PrisonerDilemmaENV(AECEnv):
    metadata = {
        "render_modes": ["human"], 
        "render_fps": CONST.RENDER_FPS, 
        "name": "PrisonerDilemma",
    }
    def __init__(self, render_mode: str=None) -> None:
        # TODO: check how to visualize the game
        self.render_mode = render_mode

        self.agents = ["agent_" + str(i) for i in range(CONST.NUM_AGENTS)]
        # self.agent_name_mapping = dict(
        #     zip(self.agents, [("A", "human"), ("A","robot"), ("B","human"), ("B","robot")])
        # )
        
        # We have 2 actions, corresponding to "cooperate" and "defect"
        self.action_spaces = {
            agent: spaces.Discrete(n=2, seed=CONST.SEED) for agent in self.agents
        }

        # The observation space is a sequence of the graph
        self._obs_space_each_round = spaces.Graph(
            node_space=spaces.Discrete(4), edge_space=spaces.Discrete(2), seed=CONST.SEED
        )
        self._observation_spaces = {
            agent: spaces.Sequence(space=self._obs_space_each_round, seed=CONST.SEED) 
                for agent in self.agents
        }
        self._initial_obs = [utils.pd_graph()]

    # @functools.lru_cache(maxsize=None)
    # def observation_space(self, agent: str) -> spaces.Space:
    #     return self._observation_spaces
    
    # @functools.lru_cache(maxsize=None)
    # def action_spaces(self) -> spaces.Space:
    #     # print(agent)

    #     return self._action_spaces

    def render(self):
        pass # TODO: implement the render function later

    def get_info(self, agent: str) -> Dict:
        # TODO: return the info
        return {}
    
    def observe(self, agent: str):
        # observation of one agent is the previous state of the other
        return self.observations[agent]
    
    def accumulate_rewards(self):
        for agent, reward in self.rewards.items():
            self._cumulative_rewards[agent] += reward
        self.rewards = {agent: 0 for agent in self.agents}
    
    def reset(self, seed:int=None, options:int=None):
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: {} for agent in self.agents} # stores action of current agent
        self.observations = {agent: self._initial_obs for agent in self.agents}
        self.number_round = 0
        """
        TODO: check: Our agent_selector utility allows easy cyclic stepping through the agents list.
        """
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    # def _observation_each_round(
    #     self, nodes: np.ndarray, edges: np.ndarray, edge_links: np.ndarray
    # ) -> spaces.graph.GraphInstance:
    #     return spaces.graph.GraphInstance(
    #         nodes=nodes, edges=edges, edge_links=edge_links,
    #     )

    def step(self, action: int):
        """
        step(action) takes in an action for the current agent (specified by
        agent_selection) and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - terminations
        - truncations
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()
        """
        if (self.terminations[self.agent_selection] or self.truncations[self.agent_selection]):
            # handles stepping an agent which is already dead
            # accepts a None action for the one agent, and moves the agent_selection to
            # the next dead agent,  or if there are no more dead agents, to the next live agent
            # self._was_dead_step(action)
            return

        '''
        the agent which stepped last had its _cumulative_rewards accounted for
        (because it was returned by last()), so the _cumulative_rewards for this
        agent should start again at 0
        '''
        # TODO: check if this is necessary
        # self._cumulative_rewards[self.agent_selection] = 0 

        # stores action of current agent
        self.state[self.agent_selection] = action

        # collect reward if it is the last agent to act
        if self._agent_selector.is_last():
            edge_links = self.observations[self.agent_selection][-1].edge_links
            partner = {f"agent_{i}": f"agent_{j}" for i, j in edge_links}

            for agent in self.agents:
                action = self.state[agent]
                partner_action = self.state[partner[agent]]
                self.rewards[agent] = CONST.REWARD_MAP[(action, partner_action)]

            self.number_round += 1
            # The truncations dictionary must be updated for all players.
            self.truncations = {
                agent: self.number_round >= CONST.MAX_ROUND for agent in self.agents
            }

            next_state = utils.pd_graph()
            # observe the current state
            edges = []
            for i, _ in edge_links:
                edges.append(self.state[f"agent_{i}"])
            self.observations[agent].append(
                spaces.graph.GraphInstance(
                    nodes=[0,1,2,3], edges=edges, edge_links=edge_links
                )
            )
            self.observations[agent].append(next_state)
        else:
            # necessary so that observe() returns a reasonable observation at all times.
            # self.state[agent] = self.observations[agent][-1]
            # # no rewards are allocated until both players give an action
            # self._clear_rewards()
            pass 

        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        # Adds .rewards to ._cumulative_rewards
        self.accumulate_rewards()
        
        if self.render_mode == "human":
            self.render()

def env(render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = PrisonerDilemmaENV(render_mode=internal_render_mode)
    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env