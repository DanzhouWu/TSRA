import numpy as np
import copy

def state2index_2(length, L, O):
    if O == 'B':
        o_index = 0
    if O == 'S':
        o_index = 1
    if O == 'I':
        o_index = 2
    if O == 'F':
        o_index = 3
    if sum(L) == 0:
        return o_index
    for i in range(length):
        if L[i] != 0:
            return 1 * 4 + o_index
    return 2 * 4 + o_index

class TSRA_AGENT():
    def __init__(self, D, arrival_rate, learning_rate, length=1):
        self.D = D
        self.arrival_rate = arrival_rate
        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.queue = [0] * self.D
        self.state = []
        self.length = length
        self.Q_table = np.zeros(shape=(3 * 4, 2), dtype=float)
        self.rho = 0

        self.initialize()

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
        self.state.append('B')
        self.state.append(copy.deepcopy(self.queue))

    def update_Q_table(self, observation):
        self.state = self.state[-4:]
        index_1 = state2index_2(length=self.length, L=self.state[-3], O=self.state[-4])
        index_2 = state2index_2(length=self.length, L=self.state[-1], O=self.state[-2])
        if observation == 'B' or observation == 'S':
            reward = 10
        elif observation == 'F' and self.action == 1: # collsion or channel
            reward = -5
        elif observation == 'I' and self.queue[0] == 1: # need to transmit
            reward =  -3
        else: # 'F' and action = 0, 'I' and no packet
            reward =  2
        self.Q_table[index_1][self.action] += self.learning_rate \
            * (reward + np.max(self.Q_table[index_2]) - self.Q_table[index_1][self.action] - self.rho)
        self.rho +=  self.learning_rate * (reward + np.max(self.Q_table[index_2]) - self.Q_table[index_1][self.action] - self.rho)

    def update_queue(self, observation):
        self.state.append(observation)
        if observation == 'S': # transmit a packet
            self.queue[self.queue.index(1)] = 0
        self.queue[:-1] = self.queue[1:]
        if  self.arrival_rate > np.random.uniform():
            self.queue[-1] = 1
        else:
            self.queue[-1] = 0
        self.state.append(copy.deepcopy(self.queue))

    def select_action(self, observation, **kwargs):
        # 0 is wait, 1 is transmit
        ###########################################
        index = state2index_2(length=self.length, L=self.queue, O=observation)

        self.epsilon *= self.epsilon_decay
        self.epsilon  = max(self.epsilon_min, self.epsilon)
        if self.epsilon > np.random.uniform():
            self.action = round(np.random.uniform())
        else:
            self.action = np.argmax(self.Q_table[index])
        if sum(self.queue) == 0:
            self.action = 0

    def update(self, observation, **kwargs):
        self.update_queue(observation)
        self.update_Q_table(observation)
        self.select_action(observation, **kwargs)

