import numpy as np
class ALOHA_AGENT():
    def __init__(self, D, arrival_rate, trans_prob):
        self.D = D
        self.arrival_rate = arrival_rate
        self.trans_prob = trans_prob
        self.queue = [0] * self.D
        self.actions_list = []
        self.initialize()

    def initialize(self):
        # initailize queue
        for i in range(self.D):
            if self.arrival_rate > np.random.uniform():
                self.queue[i] = 1

    def select_action(self, time):
        if time % self.D == 0 and time != 0: 
            self.actions_list = []
        if self.trans_prob > np.random.uniform() and sum(self.queue) != 0:
            self.action = 1
        else:
            self.action = 0

        self.actions_list.append(self.action)



    def update_queue(self,reward):
        if reward == 1 and self.action == 1: # transmit a packet
            self.queue[self.queue.index(1)] = 0
        self.queue = self.queue[1:] + [1 if self.arrival_rate > np.random.uniform() else 0]

    def update(self, reward):
        self.update_queue(reward)
