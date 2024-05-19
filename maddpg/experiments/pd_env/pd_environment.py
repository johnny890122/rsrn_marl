import functools, random
from copy import copy
import numpy as np
from typing import List, Dict, Any, Tuple
import gym.spaces as spaces
# from pettingzoo import ParallelEnv
import pd_env.CONST as CONST
import pd_env.utils as utils

class PrisonerDilemmaEnvironment():
    metadata = {
        "name": "PrisonerDilemma_environment_v0",
    }

    def __init__(self):
        """
        Initializes a new instance of the PDEnvironment class.
        """
        # Define the agents
        self.agents = [agent for agent in range(CONST.NUM_AGENTS)]
        self.agents_name_mapping = {
            agent: name for agent, name in zip(self.agents, CONST.AGENTS_NAMES)
        }
        self.n = CONST.NUM_AGENTS

        # Define the actions
        self.actions = [action for action in range(CONST.NUM_ACTIONS)]
        self.action_name_mapping = {
            action: name for action, name in zip(self.actions, CONST.ACTIONS_NAMES)
        }
        self.action_space = [spaces.Discrete(CONST.NUM_ACTIONS) for _ in self.agents]

        # Define the observation spaces
        self.observation_space = [spaces.MultiDiscrete(
            [CONST.NUM_ACTIONS for _ in range(self.n)]
        ) for _ in self.agents]
        
        # spaces.Sequence(    
            # spaces.Dict({
            #     "self_identity": spaces.Discrete(1), # self identity
            #     "agents": spaces.MultiDiscrete([CONST.NUM_AGENTS for _ in range(self.n)]), # identity of each agent
            #     "pairs": spaces.Graph(
            #         node_space=spaces.Discrete(self.n), 
            #         edge_space=spaces.Discrete(2),
            #     ),
            #     "actions": spaces.MultiDiscrete([CONST.NUM_ACTIONS for _ in range(self.n)]), # actions of each agent
            #     "payoffs": spaces.MultiDiscrete(
            #         [max(CONST.REWARD_MAP.values())+1 for _ in range(self.n)]
            #     ), # payoffs of each agent
            # })
        # ) 
        # Define other attributes
        self.timestep = None
        self.seed = None
        self.observations = None
        self.rewards = None # acuumulated payoffs
        

    def reset(self, seed: int=CONST.SEED, options: Any=None):
        """
        Resets the environment to its initial state.

        Args:
            seed (int, optional): The seed value for random number generation. Defaults to CONST.SEED.
            options (any, optional): Additional options for resetting the environment. Defaults to None.

        Returns:
            tuple: A tuple containing the observations and infos after resetting the environment.
        """
        self.timestep = 0
        self.seed = seed
        self.rewards = [0 for _ in self.agents]
        self.observations = self.initialize_round()

        # Get dummy infos. Necessary for proper parallel_to_aec conversion
        infos = {agent: self.info(agent) for agent in self.agents}

        return self.observations
    
    def observe(self, agent: int, moment: int) -> Tuple[Any]:
            """
            Retrieve the observation for a specific agent at a given moment.

            Args:
                agent (int): The index of the agent.
                moment (int): The moment in time.

            Returns:
                Tuple[Any]: The observation for the specified agent at the given moment.
            """
            return self.observations[moment][agent]
    
    def observe_all(self, agent: int) -> List[Tuple[Any]]:
            """
            Returns a list of observations for the specified agent at each moment in time.

            Args:
                agent (int): The index of the agent.

            Returns:
                List[Tuple[Any]]: A list of observations, where each observation is a tuple.
            """
            return [self.observe(agent, moment) for moment in range(self.timestep+1)]

    def step(self, actions: List[int], episode_step: int):
        """
        Perform a step in the prisoner's dilemma environment.

        Args:
            actions (List[int]): A list of actions chosen by each agent.

        Returns:
            Tuple: A tuple containing the following elements:
                - observations (List[List[List[int]]]): A list of observations for each agent at each timestep.
                - rewards (List[int]): A list of accumulated rewards for each agent.
                - terminations (Dict[int, bool]): A dictionary indicating whether each agent has terminated.
                - truncations (Dict[int, bool]): A dictionary indicating whether each agent has been truncated.
                - infos (Dict[int, Any]): A dictionary containing additional information for each agent.
        """
        actions = [np.argmax(action) for action in actions]
        # Collect payoffs of current round
        self.payoffs = []
        pairs = utils.pd_pairs(self.agents)
        for agent in self.agents:
            obersevation = self.observe(agent, moment=self.timestep)
            partner = utils.my_partner(agent, pairs)

            payoff = CONST.REWARD_MAP[
                (actions[agent], actions[partner])
            ]
            self.payoffs.append(payoff)

            # # Update the current observation
            # self.observations[self.timestep][agent][3] = actions
            # self.observations[self.timestep][agent][4] = payoffs

        # Update the rewards
        for agent in self.agents:
            self.rewards[agent] += self.payoffs[agent]

        # Initialize a new round if the maximum number of rounds has not been reached
        if self.timestep <= CONST.MAX_ROUNDS:
            new_round = self.initialize_round()
            self.observations.append(new_round)

        # Check termination conditions
        # terminations = {agent: False for agent in self.agents}
        
        # Check truncation conditions (overwrites termination conditions)
        if self.timestep > CONST.MAX_ROUNDS:
            truncations = {agent: True for agent in self.agents}
        else:
            truncations = {agent: False for agent in self.agents}

        # Get dummy infos (not used in this example)
        infos = {agent: self.info(agent) for agent in self.agents}

        self.timestep += 1
        
        return self.observations, self.rewards, truncations, infos

    def render(self):
        """Renders the environment."""
        pass

    # def observation_space(self, agent):
    #     return self.observation_spaces[agent]

    # @functools.lru_cache(maxsize=None) # Cache the action space
    # def action_space(self, agent) -> spaces.Space:
    #     # lru_cache allows action spaces to be memoized, reducing clock cycles required to get each agent's space.
    #     return spaces.Discrete(CONST.NUM_ACTIONS)

    def info(self, agent: int) -> Dict:
        info = {} # TODO: return the info
        return info

    def initialize_round(self):
        pairs = utils.pd_pairs(self.agents)
        actions = [CONST.COOPERATE for _ in self.agents]
        payoffs = [0 for _ in self.agents]

        state = []
        for agent in self.agents:
            state.append(np.array(actions))
            # [
            #     agent, # self identity 
            #     self.agents, # identity of each agent
            #     pairs, # pairs of agents
            #     actions, # actions of each agent
            #     payoffs, # payoffs of each agent
            # ]

        return state