import numpy as np
from tqdm import tqdm
from aloha_agent import ALOHA_AGENT
from fsra_agent import FSRA_AGENT
from environment import ENVIRONMENT

# get FSRA timelt throughput
def FSRA(D, D_, pb1, pt1, ps1, pb2, ps2, iteration=int(1e7)):
    aloha = ALOHA_AGENT(D=D, arrival_rate=pb1, trans_prob=pt1)
    aloha.initialize()

    FSRA_agent = FSRA_AGENT(D=D_, arrival_rate=pb2, learning_rate=0.01)
    FSRA_agent.initailize()

    env = ENVIRONMENT(aloha_channel=ps1, agent_channel=ps2, aloha=aloha, agent=FSRA_agent)

    FSRA_reward = []
    for _ in tqdm(range(iteration)):
        aloha_reward, agent_reward ,observation = env.step()
        env.aloha.update(observation=observation)
        env.agent.update(observation=observation)
        FSRA_reward.append(aloha_reward + agent_reward)

    FSRA_timely_throughput = np.mean(FSRA_reward[-int(1e5):])
    print('FSRA_timely_throughput:', FSRA_timely_throughput)

if __name__ == "__main__":
    pb1 = 0.5
    ps1 = 0.7
    pt1 = 0.4

    pb2 = 0.4
    ps2 = 0.6

    D = 5
    D_= 5
    FSRA(D=D, D_=D_, pb1=pb1, pt1=pt1, ps1=ps1, pb2=pb2, ps2=ps2)
