import numpy as np
from tqdm import tqdm
from multichainLP import multichainLP 
from aloha_agent import ALOHA_AGENT
from environment import ENVIRONMENT
from specify_agent import SPECIFY_AGENT

# get upper_bound timely throughput
def upper_bound(D, D_, pb1, pt1, ps1, pb2, ps2, iteration=int(1e5)):

    aloha = ALOHA_AGENT(D=D, arrival_rate=pb1, trans_prob=pt1)
    aloha.initialize()

    # get LP agent policy
    LP_policy = multichainLP(D=D, D_=D_, pb1=pb1, pt1=pt1, ps1=ps1, pb2=pb2, ps2=ps2)

    sp_agent = SPECIFY_AGENT(D=D_, arrival_rate=pb2, policy=LP_policy)
    sp_agent.initialize()

    env = ENVIRONMENT(aloha_channel=ps1, agent_channel=ps2, aloha=aloha, agent=sp_agent)

    UP_reward = []
    for _ in tqdm(range(iteration)):
        aloha_reward, agent_reward ,observation = env.step()
        env.aloha.update(observation=observation)
        env.agent.update(observation=observation, aloha_queue=env.aloha.queue)
        UP_reward.append(aloha_reward + agent_reward)

    Upper_bound_timely_throughput = np.mean(UP_reward)
    print('Upper_bound_timely_throughput:', Upper_bound_timely_throughput)

if __name__ == '__main__':
    pb1 = 0.5
    ps1 = 0.7
    pt1 = 0.4

    pb2 = 0.4
    ps2 = 0.6

    D = 5
    D_= 5

    upper_bound(D=D, D_= D_, pb1=pb1, pt1=pt1, ps1=ps1, pb2=pb2, ps2=ps2)
