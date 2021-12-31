import numpy as np

class ENVIRONMENT():
    def __init__(self, channels, agent_list):
        self.channels = channels
        self.agent_list = agent_list
        self.N = len(agent_list)
        # self.recoder = {'S':0, 'I':0, 'C':0, 'E':0}

    def step(self, **kargs):
        rewards = 0
        action = [agent.action for agent in self.agent_list]

        energy = sum(action)
        assert energy >= 0 

        if energy == 0: 
            observations = ['I'] * self.N # idle
        elif energy > 1: 
            observations = ['F'] * self.N # collided
        else:
            idx = action.index(1)
            if self.channels[idx] > np.random.uniform():
                rewards = 1
                observations = ['B'] * self.N # busy
                observations[idx] = 'S' # successful
            else:
                observations = ['F'] * self.N
        return rewards, energy, observations