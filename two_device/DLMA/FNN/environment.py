import numpy as np
class ENVIRONMENT(object):
    def __init__(self, aloha_channel, agent_channel, aloha, agent):
        self.aloha_channel = aloha_channel
        self.agent_channel = agent_channel
        self.aloha = aloha
        self.agent = agent

    def step(self):
        aloha_reward, agent_reward = 0, 0
        if self.aloha.action == 0: # aloha waits
            if self.agent.action == 0: # agent waits
                observation = 'I'
            elif self.agent_channel > np.random.uniform(): # agents transmit successfully
                agent_reward = 1
                observation = 'S'
            else: # agents channel error
                observation = 'F'
        else: # aloha transmits
            if self.agent.action == 0: # agent waits
                if self.aloha_channel > np.random.uniform(): # aloha transmits successfully
                    aloha_reward = 1
                    observation = 'B' 
                else:
                    observation = 'F' # aloha channel error
            else:
                observation = 'F' # collision
                
        return aloha_reward, agent_reward, observation
