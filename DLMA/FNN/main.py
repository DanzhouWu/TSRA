import psutil
import os
import time
import numpy as np
from tqdm import tqdm
from aloha_agent import ALOHA_AGENT
from DQN_agent import DQN_AGENT
from environment import ENVIRONMENT
from until import return_action, return_observation

def DLMA_FNN(D, D_, pb1, pt1, ps1, pb2, ps2, iteration=int(1e5)):
    aloha = ALOHA_AGENT(D=D, arrival_rate=pb1, trans_prob=pt1)
    aloha.initialize()

    DLMA = DQN_AGENT(D=D_, arrival_rate=pb2, 
                    state_size=160,
                    n_actions=2,
                    n_nodes=2,
                    memory_size=1000,
                    replace_target_iter=20,
                    batch_size=64,
                    learning_rate=0.01,
                    gamma=0.9,
                    epsilon=1,
                    epsilon_min=0.005,
                    epsilon_decay=0.995,
                    alpha=0
                    )

    DLMA.initailize()

    env = ENVIRONMENT(aloha_channel=ps1, agent_channel=ps2, aloha=aloha, agent=DLMA)
    state = [0] * DLMA.state_size

    DLMA_FNN_reward = []

    begin = time.time()
    for i in tqdm(range(iteration)):
        aloha_reward, agent_reward ,observation = env.step()
        env.aloha.update(observation)
        env.agent.update(observation, state)
        DLMA_FNN_reward.append(aloha_reward + agent_reward)

        next_state = state[8:] + return_action(env.agent.action) + return_observation(observation) + [agent_reward, aloha_reward]

        env.agent.store_transition(state, env.agent.action, agent_reward, aloha_reward, next_state)
        if i > 100 and (i % 5 == 0):
            env.agent.learn()       # internally iterates default (prediction) model
        state = next_state

    DLMA_FNN_timely_throughput = np.mean(DLMA_FNN_reward)
    print('DLMA_FNN_timely_throughput:', DLMA_FNN_timely_throughput)

    end = time.time()
    print(u'当前进程的运行时间: ' , (end - begin), 's')
    print(u'当前进程的内存使用：%.4f MB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024))

if __name__ == '__main__':
    pb1 = 0.5
    ps1 = 0.7
    pt1 = 0.4

    pb2 = 0.4
    ps2 = 0.6

    D = 2
    D_= 2

    DLMA_FNN(D=D, D_= D_, pb1=pb1, pt1=pt1, ps1=ps1, pb2=pb2, ps2=ps2)
