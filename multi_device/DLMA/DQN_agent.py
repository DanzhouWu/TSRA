import numpy as np
import copy
import tensorflow as tf
from tensorflow.python.keras import backend as K

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Add, Activation
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.activations import softmax

class DQN_AGENT(object):
    def __init__(self,
                D, 
                arrival_rate, 
                state_size, 
                n_actions,
                n_nodes,
                memory_size, 
                replace_target_iter,
                batch_size,
                learning_rate=0.01,
                gamma=0.9,
                epsilon=1,
                epsilon_min=0.01,
                epsilon_decay=0.995,
                alpha=0):

        self.D = D
        self.arrival_rate = arrival_rate
        self.state_size = state_size
        self.memory_size = memory_size
        self.replace_target_iter = replace_target_iter
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha
        self.n_nodes = n_nodes
        self.n_actions = n_actions

        self.memory = np.zeros((self.memory_size, self.state_size * 2 + (self.n_nodes+1))) # memory_size * len(s, a, r1, r2, s_)
        self.learn_step_counter = 0
        self.memory_couter = 0

        ### build mode
        self.model        = self.build_ResNet_model() # model: evaluate Q value
        self.target_model = self.build_ResNet_model() # target_mode: target network

        self.queue = [0] * self.D

        self.initailize()

    def initailize(self):
        # initailize queue
        for i in range(self.D):
            if self.arrival_rate > np.random.uniform():
                self.queue[i] = 1


    def alpha_function(self, action_values):
        action_values_list = []
        if self.alpha == 1:
            action_values_list = [np.log(action_values[2*j]) + np.log(action_values[2*j+1])  for j in range(self.n_actions)]
        elif self.alpha == 0:
            action_values_list = [action_values[2*j] + action_values[2*j+1] for j in range(self.n_actions)]
        elif self.alpha == 100:
            action_values_list = [min(action_values[2*j], action_values[2*j+1]) for j in range(self.n_actions)]
        else:
            action_values_list = [1/(1-self.alpha) * (action_values[2*j]**(1-self.alpha) + \
                                    action_values[2*j+1]**(1-self.alpha)) for j in range(self.n_actions)]
        return np.argmax(action_values_list)


    def build_ResNet_model(self):
        inputs = Input(shape=(self.state_size, ))
        h1 = Dense(64, activation="relu", kernel_initializer=he_normal(seed=247))(inputs) #h1
        h2 = Dense(64, activation="relu", kernel_initializer=he_normal(seed=2407))(h1) #h2

        outputs =  Dense(self.n_actions*self.n_nodes, kernel_initializer=he_normal(seed=27))(h2)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss='mse', optimizer='rmsprop')
        return model

    def choose_action(self, state):
        state = state[np.newaxis, :]
        self.epsilon *= self.epsilon_decay
        self.epsilon  = max(self.epsilon_min, self.epsilon)
        
        if np.random.uniform(0, 1) < self.epsilon:
            self.action =  np.random.randint(0, self.n_actions)
        else:
            action_values = self.model.predict(state)
            self.action = self.alpha_function(action_values[0])
        if sum(self.queue) == 0:
            self.action = 0


    def store_transition(self, s, a, r1, r2, s_): # s_: next_state
        if not hasattr(self, 'memory_couter'):
            self.memory_couter = 0
        transition = np.concatenate((s, [a, r1, r2], s_))
        index = self.memory_couter % self.memory_size
        self.memory[index, :] = transition
        self.memory_couter   += 1

    def repalce_target_parameters(self):
        weights = self.model.get_weights()
        self.target_model.set_weights(weights)

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.repalce_target_parameters() # iterative target model
        self.learn_step_counter += 1

        if self.memory_couter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_couter, size=self.batch_size)        
        batch_memory = self.memory[sample_index, :]

        state      = batch_memory[:, :self.state_size]
        action     = batch_memory[:, self.state_size].astype(int) # float -> int
        reward1    = batch_memory[:, self.state_size+1]
        reward2    = batch_memory[:, self.state_size+2]
        next_state = batch_memory[:, -self.state_size:]

        q = self.model.predict(state) # state		
        q_targ = self.target_model.predict(next_state) # next state

        for i in range(self.batch_size):
            action_ = self.alpha_function(q_targ[i])
            q[i][2*action[i]]   = reward1[i] + q_targ[i][2*action_]
            q[i][2*action[i]+1] = reward2[i] + q_targ[i][2*action_+1]
 
        self.model.fit(state, q, self.batch_size, epochs=1, verbose=0)

    def update_queue(self, observation):
        if observation == 'S': # transmit a packet
            self.queue[self.queue.index(1)] = 0
        self.queue[:-1] = self.queue[1:]
        if  self.arrival_rate > np.random.uniform():
            self.queue[-1] = 1
        else:
            self.queue[-1] = 0

    def update(self, observation, state):
        self.update_queue(observation)
        self.choose_action(np.array(state))