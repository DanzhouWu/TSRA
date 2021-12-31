import random
import numpy as np
import math
import copy

class Learn2MAC():
    def __init__(self, N, l, d, eta, T, arrival_rate, channels):
        self.N = N # N slots, the deadline of delay-constrained
        self.l = l # valid lenght, the lenght is 1 in delay-constrained
        self.d = d # dictionary size
        self.eta = eta 
        self.T = T
        self.arrival_rate = arrival_rate
        self.channels = channels
        self.alpha = math.sqrt(2 / (T * max(1, math.pow(self.eta*self.N, 2))))
        
        self.pattern_dict = [[0]*self.N]

        self.queue = [0] * self.N
        self.pre_queue = []
        self.actions_list = []
        self.rewards_list = []

        self.initialize()

    def _init_pattern_dictt(self):
        while len(self.pattern_dict) < min(self.d, math.pow(2, self.N)):
            n = random.randint(self.l, self.N)
            tmp = [1] * n + [0] * (self.N - n)
            random.shuffle(tmp)
            if tmp not in self.pattern_dict:
                self.pattern_dict.append(tmp)
        self.d = len(self.pattern_dict)
        self.p = [1/self.d] * self.d
            # print(self.pattern_dict)

    def _select_pattern(self):
        tmp = np.random.uniform()
        assert abs(sum(self.p) - 1) < 0.0001
        for i in range(self.d):
            if sum(self.p[:i+1]) >= tmp:
                self.pattern = self.pattern_dict[i]
                return

    def initialize(self):
        self._init_pattern_dictt()
        self._select_pattern()
        for i in range(self.N):
            if self.arrival_rate > np.random.uniform():
                self.queue[i] = 1

    def select_action(self, time):
        idx = time % self.N 
        
        self.action = 1 if self.pattern[idx] == 1 and 1 in self.queue else 0
        
        self.actions_list.append(self.action)

    def _update_queue_bernouli(self):
        self.queue = self.queue[1:] + [1 if self.arrival_rate > np.random.uniform() else 0]

    def _adjust_R(self, pattern, pre_queue, others):
        for slot_idx in range(self.N):
            assert others[slot_idx] >= 0
            if pattern[slot_idx] == 1 and 1 in pre_queue and others[slot_idx] == 0 \
                and self.channels > np.random.uniform():
                return 1
            else:
                pre_queue = pre_queue[1:] + [1 if self.arrival_rate > np.random.uniform() else 0]
        return 0

    def _update_p(self, patterns):
        v = []
        denominator = 0
        others = [patterns[i] - self.actions_list[i] for i in range(self.N)]
        for idx, pattern in enumerate(self.pattern_dict):
            # pre_queue = copy.deepcopy(self.pre_queue)
            pre_queue = copy.deepcopy(self.queue)
            R_pi = self._adjust_R(pattern, pre_queue, others)
            # R_pi = 0
            # for i in range(self.N):
            #     if pattern[i] == 1 and self.pre_queue[i] == 1 and others[i] == 0:
            #         if self.channels > np.random.uniform(): 
            #             R_pi = 1
            #             break
            v_tmp = R_pi - self.eta * sum(pattern)
            v.append(v_tmp)
            denominator += self.p[idx] * math.exp(-1 * self.alpha * v_tmp)
            
        for idx, pattern in enumerate(self.pattern_dict):
            self.p[idx] = self.p[idx] * math.exp(-1 * self.alpha * v[idx]) / denominator
        
        self.pre_queue = []
        self.actions_list = []
        self.rewards_list = []

    def update(self, time, reward, patterns):
        idx = (time + 1) % self.N

        self.rewards_list.append(reward)
        self.pre_queue.append(self.queue[0])

        if reward == 1 and self.action == 1:
            assert 1 in self.queue
            self.queue[self.queue.index(1)] = 0
        
        if idx == 0:
            self._update_p(patterns)
            self._select_pattern()
        
        self._update_queue_bernouli()
