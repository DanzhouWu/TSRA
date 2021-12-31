import os, time, psutil
import numpy as np
from tqdm import tqdm
from learn2mac import Learn2MAC
from aloha_agent import ALOHA_AGENT
from environment import Environment

def main(n, D, l, d, eta, pb1, pt1, ps1, pb2, ps2, T=int(1e5)):

    begin = time.time()
    agents_list = [ALOHA_AGENT(D, arrival_rate=pb1, trans_prob=pt1), \
        Learn2MAC(D, l, d, eta, T, pb2, ps2)]

    env = Environment(agents_list=agents_list, channels=[ps1, ps2])

    reward_list = []
    energy_list = []
    for t in tqdm(range(T)):
        for agent in agents_list: 
            agent.select_action(t)
        
        reward, energy, patterns = env.step(t)
        reward_list.append(reward)
        energy_list.append(energy)

        for agent in agents_list: 
            if isinstance(agent, ALOHA_AGENT): agent.update(reward)       # aloha
            else: agent.update(t, reward, patterns)                    # learn2mac

    end = time.time()
    print('Throu = {}'.format(np.mean(reward_list[-int(1e4):])))
    print('Energy = {}'.format(np.mean(energy_list[-int(1e4):])))
    print('Time = {}s'.format(end-begin))
    print('Memory = {}MB'.format(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024))
        

if __name__ == '__main__':
    pb1 = 0.5
    ps1 = 0.7
    pt1 = 0.4
    pb2 = 0.6
    ps2 = 0.4
    
    n = 2 # user number
    D = 5 # deadline
    l = 1 # valid length

    eta = 1 / D
    d = 100 # dictionary size
    throughput_list = []
    power_list = []

    main(n, D, l, d, eta, pb1, pt1, ps1, pb2, ps2)