import numpy as np
from tqdm import tqdm
import random
import argparse
from tsra_agent import TSRA_AGENT
from environment import ENVIRONMENT
from aloha_agent import ALOHA_AGENT

random.seed(20)
np.random.seed(1)

def main(N1, N2, D, parameter, iteration=int(1e5)):

    agent_list = [TSRA_AGENT(D=D, arrival_rate=parameter[i], learning_rate=0.01, length=1) \
        for i in range(N2)] # parameterss pb2
    n1_list = [ALOHA_AGENT(D=D, arrival_rate=1, trans_prob=1/(4*N1)) for _ in range(N1)]
    agent_list.extend(n1_list)

    channels = list(parameter[N2:]) # parameters ps2
    n1_channels = [0.5 for _ in range(N1)]
    channels.extend(n1_channels)
    
    env = ENVIRONMENT(channels=channels, agent_list=agent_list)

    reward_list = []
    energy_list = []

    for time in tqdm(range(iteration)):
        reward, energy, observations = env.step(time=time) 
        
        for i in range(N2): 
            env.agent_list[i].update(observation=observations[i], time=time, N=N2)
        
        for i in range(N1): 
            env.agent_list[N2+i].update(observation=observations[N2+i])

        reward_list.append(reward)
        energy_list.append(energy)
    
    throughput, power = np.mean(reward_list[-int(1e4):]), np.mean(energy_list[-int(1e4):]) 
    print('Throu = {}'.format(throughput))
    print('Energy = {}'.format(power))
    return throughput, power

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n1", "--n1",type=int, default=3)
    parser.add_argument("-n2", "--n2",type=int, default=100)
    parser.add_argument("-D", "--D",type=int, default=10)
    parser.add_argument("-idx", "--idx",type=int, default=0)
    args = parser.parse_args()
    
    n1 = args.n1
    n2 = args.n2
    D = args.D
    idx = args.idx

    case_num = 100

    parameters = np.load('../RandomParameters_seed_'+ str(n2) +'_100.npy')
    throughput_list = []
    power_list = []
    for parameter in parameters[case_num*idx: case_num*(idx+1)]:

        throughput, power = main(n1, n2, D, parameter)

        throughput_list.append(throughput)
        power_list.append(power)

    throughput_array = np.array(throughput_list).reshape((case_num, 1))
    power_array = np.array(power_list).reshape((case_num, 1))
    filepath = 'TSRA' + 'D_' + str(D) + '_n1_' + str(n1) + '_n2_' + str(n2) + '_idx_' + str(idx)
    np.save(filepath + 'throuput', throughput_array)
    np.save(filepath + 'power', power_array)
    print('Finish n1={} n2={} D={} idx={}'.format(n1, n2, D, idx))