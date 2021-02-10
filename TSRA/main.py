import psutil
import os
import time
import numpy as np
from aloha_agent import ALOHA_AGENT
from tsra_agent import TSRA_AGENT
from environment import ENVIRONMENT
from tqdm import tqdm

def TSRA(D, D_, pb1, pt1, ps1, pb2, ps2, iteration=int(1e5)):
    aloha = ALOHA_AGENT(D=D, arrival_rate=pb1, trans_prob=pt1)
    aloha.initialize()

    TSRA_agent = TSRA_AGENT(D=D_, arrival_rate=pb2, learning_rate=0.01, gamma=0.9, length=1)
    TSRA_agent.initailize()

    env = ENVIRONMENT(aloha_channel=ps1, agent_channel=ps2, aloha=aloha, agent=TSRA_agent)
    TSRA_reward = []

    begin = time.time()
    for _ in tqdm(range(iteration)):
        aloha_reward, agent_reward ,observation = env.step()
        env.aloha.update(observation=observation)
        env.agent.update(observation=observation)
        TSRA_reward.append(aloha_reward + agent_reward)

    TSRA_timely_throughput = np.mean(TSRA_reward)
    print('TSRA_timely_throughput:', TSRA_timely_throughput)

    end = time.time()
    print('time: ' , (end - begin), 's')
    print('memory: %.4f MB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024) )

if __name__ == "__main__":
    pb1 = 0.5
    ps1 = 0.7
    pt1 = 0.4

    pb2 = 0.4
    ps2 = 0.6

    D = 10
    D_= 10
    TSRA(D=D, D_=D_, pb1=pb1, pt1=pt1, ps1=ps1, pb2=pb2, ps2=ps2)
