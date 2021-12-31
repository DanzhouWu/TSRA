import psutil
import os
import time
import numpy as np
from tqdm import tqdm
from aloha_agent import ALOHA_AGENT
from DQN_brain import DQN
from environment import ENVIRONMENT
from until import return_action, return_observation

def DLMA_RNN(D, D_, pb1, pt1, ps1, pb2, ps2, iteration=int(1e5)):
    aloha = ALOHA_AGENT(D=D, arrival_rate=pb1, trans_prob=pt1)
    aloha.initialize()

    DLMA = DQN(D=D_, arrival_rate=pb2, 
                    features=8,
                    n_actions=2,
                    n_nodes=2,
                    state_length=4,
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

    channel_state = [0] * DLMA.features
    state = np.zeros((4, len(channel_state)))

    DLMA_RNN_reward = []
    begin = time.time()
    for i in tqdm(range(iteration)):
        state = np.vstack([state[1:], channel_state])
        aloha_reward, agent_reward ,observation = env.step()
        env.aloha.update(observation)
        env.agent.update(observation, state)


        DLMA_RNN_reward.append(aloha_reward + agent_reward)
        next_channel_state = return_action(env.agent.action) + return_observation(observation) + [agent_reward, agent_reward]
        experience = np.concatenate([channel_state, [env.agent.action, agent_reward, agent_reward], next_channel_state])

        env.agent.add_experience(experience)

        if i > 100 and (i % 5 == 0):
            env.agent.learn()       # internally iterates default (prediction) model
        channel_state = next_channel_state

    DLMA_RNN_timely_throughput = np.mean(DLMA_RNN_reward)
    print('DLMA_RNN_timely_throughput:', DLMA_RNN_timely_throughput)

    end = time.time()
    print('time: ' ,(end - begin), 's')
    print('memory: %.4f MB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024))

if __name__ == '__main__':
    pb1 = 0.5
    ps1 = 0.7
    pt1 = 0.4

    pb2 = 0.4
    ps2 = 0.6

    D = 2
    D_= 2

    DLMA_RNN(D=D, D_= D_, pb1=pb1, pt1=pt1, ps1=ps1, pb2=pb2, ps2=ps2)