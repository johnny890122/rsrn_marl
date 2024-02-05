import numpy as np
from gymnasium import spaces
import CONST
from typing import List

def pd_edge_links() -> List[List[int]]:
    first_pair = np.random.choice(CONST.NODES, 2, replace=False).tolist()
    second_pair = list(set(CONST.NODES) - set(first_pair))
    return [first_pair, first_pair[::-1], second_pair, second_pair[::-1]]

def pd_graph() -> spaces.Space:
    return spaces.graph.GraphInstance(
        nodes=CONST.NODES, edges=[CONST.NONE]*CONST.NUM_AGENTS, edge_links=pd_edge_links()
    )
