import numpy as np

class ENVIRONMENT():
    def __init__(self, channels, agent_list):
        self.channels = channels
        self.agent_list = agent_list
        self.N = len(agent_list)
        self.recoder = {'S':0, 'I':0, 'C':0, 'E':0}
        
    def step(self, **kargs):
        rewards = [0] * self.N
        action = [0] * self.N
        
        for i in range(self.N): 
            action[i] = self.agent_list[i].action

        energy = sum(action)
        assert energy >= 0 

        if energy == 0: 
            observations = ['I'] * self.N # idle
            if kargs['time'] >= 9e4: self.recoder['I'] += 1
        elif energy > 1: 
            observations = ['F'] * self.N # collided
            if kargs['time'] >= 9e4: self.recoder['C'] += 1
        else:
            idx = action.index(1)
            if self.channels[idx] > np.random.uniform():
                rewards[idx] = 1
                observations = ['B'] * self.N # busy
                observations[idx] = 'S' # successful
                if kargs['time'] >= 9e4: self.recoder['S'] += 1
            else:
                observations = ['F'] * self.N
                if kargs['time'] >= 9e4: self.recoder['E'] += 1

        return sum(rewards), energy, observations