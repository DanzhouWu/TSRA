import numpy as np
class ALOHA_AGENT(object):
    def __init__(self, D, arrival_rate, trans_prob):
        self.D = D
        self.arrival_rate = arrival_rate
        self.trans_prob = trans_prob
        self.queue = [0] * self.D
    
    def initialize(self):
        # initailize queue
        for i in range(self.D):
            if self.arrival_rate > np.random.uniform():
                self.queue[i] = 1
        # initailize action
        # 0 is wait, 1 is transmit
        if self.trans_prob > np.random.uniform() and sum(self.queue) != 0:
            self.action = 1
        else:
            self.action = 0
    
    def select_action(self):
        if self.trans_prob > np.random.uniform() and sum(self.queue) != 0:
            self.action = 1
        else:
            self.action = 0

    def update_queue(self,observation):
        if observation == 'S': # transmit a packet
            self.queue[self.queue.index(1)] = 0
        self.queue[:-1] = self.queue[1:]
        if  self.arrival_rate > np.random.uniform():
            self.queue[-1] = 1
        else:
            self.queue[-1] = 0
    
    def update(self,observation):
        self.update_queue(observation)
        self.select_action()
