
SEED = 42
RENDER_FPS = 4

NUM_AGENTS = 4
MAX_ROUND = 2

COOPERATE = 0
DEFECT = 1
NONE = 2

NODES = [0,1,2,3]
REWARD_MAP = {
    (COOPERATE, COOPERATE): 3, 
    (COOPERATE, DEFECT): 0,
    (DEFECT, COOPERATE): 5,
    (DEFECT, DEFECT): 1,
}