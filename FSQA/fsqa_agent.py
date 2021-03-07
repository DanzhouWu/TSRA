from until import return_queue_index, state2index_
import numpy as np
import copy

class FSQA_AGENT(object):
    def __init__(self, D, arrival_rate, learning_rate, gamma):
        self.D = D
        self.arrival_rate = arrival_rate
        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.queue = [0] * self.D
        self.state = []
        self.Q_table = np.zeros(shape=(2**self.D*4, 2), dtype=float)

    def initailize(self):
        # initailize queue
        for i in range(self.D):
            if self.arrival_rate > np.random.uniform():
                self.queue[i] = 1
        # initailize action
        if 0.5 > np.random.uniform() and sum(self.queue) != 0:
            self.action = 1
        else:
            self.action = 0
        self.state.append('I')
        self.state.append(copy.deepcopy(self.queue))

    def update_Q_table(self, observation):
        index_1 = state2index_(L2=self.state[-3], O=self.state[-4])
        index_2 = state2index_(L2=self.state[-1], O=self.state[-2])
        if observation == 'B' or observation == 'S':
            reward = 1
        else:
            reward = 0
        self.Q_table[index_1][self.action] += self.learning_rate \
            * (reward + self.gamma * np.max(self.Q_table[index_2]) - self.Q_table[index_1][self.action])

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

    def select_action(self, observation):
        # 0 is wait, 1 is transmit
        ###########################################
        index = state2index_(L2=self.queue, O=observation)

        self.epsilon *= self.epsilon_decay
        self.epsilon  = max(self.epsilon_min, self.epsilon)
        if self.epsilon > np.random.uniform():
            self.action = round(np.random.uniform())
        else:
            self.action = np.argmax(self.Q_table[index])
        if sum(self.queue) == 0:
            self.action = 0

    def update(self, observation):
        self.update_queue(observation)
        self.update_Q_table(observation)
        self.select_action(observation)
