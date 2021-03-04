from until import return_queue_index, state2index

import numpy as np
import copy
class SPECIFY_AGENT(object):
    def __init__(self, D, arrival_rate, policy):
        self.D = D
        self.arrival_rate = arrival_rate
        self.queue = [0] * self.D
        self.policy = policy

    def initialize(self):
        # initailize queue
        for i in range(self.D):
            if self.arrival_rate > np.random.uniform():
                self.queue[i] = 1
        # initailize action
        if 0.5 > np.random.uniform() and sum(self.queue) != 0:
            self.action = 1
        else:
            self.action = 0

    def update_queue(self, observation):
        if observation == 'S': # transmit a packet
            self.queue[self.queue.index(1)] = 0
        self.queue[:-1] = self.queue[1:]
        if  self.arrival_rate > np.random.uniform():
            self.queue[-1] = 1
        else:
            self.queue[-1] = 0

    def select_action(self, observation, aloha_queue):
        # 0 is wait, 1 is transmit
        ###########################################
        index = state2index(L1=aloha_queue, L2=self.queue, O=observation)

        self.action = np.argmax(self.policy[index,:])
        if sum(self.queue) == 0:
            self.action = 0
        if np.np.argmax(self.policy[index,0]) > np.random.uniform():
            self.action = 0
        else:
            self.action = 1
        if sum(self.queue) == 0:
            self.action = 0
    def update(self, observation, aloha_queue):
        self.update_queue(observation)
        self.select_action(observation, aloha_queue)
