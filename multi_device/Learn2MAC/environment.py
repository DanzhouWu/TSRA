import numpy as np
from utils import merge_pattern
class Environment():
    def __init__(self, agents_list, channels):
        self.agents_list = agents_list
        self.channels = channels
        self.N = len(agents_list) # users number
        self.recoder = {'S':0, 'I':0, 'C':0, 'E':0}
        self.D = self.agents_list[1].N # deadline

    def step(self, time):
        reward = 0
        agent_idx = 0
        actions_list = [agent.action for agent in self.agents_list]
        energy = sum(actions_list)
        patterns = []
        assert energy >= 0 
        if energy == 1:
            assert 1 in actions_list
            agent_idx = actions_list.index(1)
            if self.channels[agent_idx] > np.random.uniform():
                reward = 1
        
        if (time+1) % self.D == 0: patterns = merge_pattern(self.agents_list) # useful for update p
        return reward, energy, patterns
