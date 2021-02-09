import numpy as np
from tqdm import tqdm
from aloha_agent import ALOHA_AGENT
from fsqa_agent import FSQA_AGENT
from environment import ENVIRONMENT

# # get FSQA timelt throughput
def FSQA(D, D_, pb1, pt1, ps1, pb2, ps2, iteration=int(1e5)):
    aloha = ALOHA_AGENT(D=D, arrival_rate=pb1, trans_prob=pt1)
    aloha.initialize()

    FSQA_agent = FSQA_AGENT(D=D_, arrival_rate=pb2, learning_rate=0.01, gamma=0.99)
    FSQA_agent.initailize()

    FSQA_reward = []
    env = ENVIRONMENT(aloha_channel=ps1, agent_channel=ps2, aloha=aloha, agent=FSQA_agent)

    for _ in tqdm(range(iteration)):
        aloha_reward, agent_reward ,observation = env.step()
        env.aloha.update(observation=observation)
        env.agent.update(observation=observation)
        FSQA_reward.append(aloha_reward + agent_reward)

    FSQA_timely_throughput = np.mean(FSQA_reward)
    print('FSQA_timely_throughput:', FSQA_timely_throughput)

if __name__ == "__main__":
    pb1 = 0.5
    ps1 = 0.7
    pt1 = 0.4

    pb2 = 0.4
    ps2 = 0.6

    D = 2
    D_= 2
    FSQA(D=D, D_=D_, pb1=pb1, pt1=pt1, ps1=ps1, pb2=pb2, ps2=ps2)
