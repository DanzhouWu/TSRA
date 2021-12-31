import random
import numpy as np
from tqdm import tqdm
from learn2mac import Learn2MAC
from environment import Environment
import argparse

random.seed(20)
np.random.seed(1)

def main(n2, D, l, d, eta, parameter, T=int(1e5)):
    agents_list = [Learn2MAC(D, l, d, eta, T, parameter[i], parameter[i+n2]) for i in range(n2)]
    
    env = Environment(agents_list=agents_list, channels=parameter[n2:])

    reward_list = []
    energy_list = []
    for time in tqdm(range(T)):
        for agent in agents_list:
            agent.select_action(time)

        reward, energy, patterns = env.step(time)
        reward_list.append(reward)
        energy_list.append(energy)

        for agent in agents_list: 
            agent.update(time, reward, patterns)  

    throughput, power = np.mean(reward_list[-int(1e4):]), np.mean(energy_list[-int(1e4):]) 
    print('Throu = {}'.format(throughput))
    print('Energy = {}'.format(power))
    return throughput, power

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-D", "--D", type=int, default=10)
    parser.add_argument("-n2", "--n2", type=int, default=10)
    parser.add_argument("-idx", "--idx", type=int, default=0)
    args = parser.parse_args()
    
    D = args.D
    n2 = args.n2
    idx = args.idx
    eta = 1 / D
    d = 100 # dictionary size
    l = 1 # valid length

    case_num = 100

    parameters = np.load('../RandomParameters_seed_10_100.npy')
    throughput_list = []
    power_list = []
    for parameter in parameters[case_num*idx: case_num*(idx+1)]:

        throughput, power = main(n2, D, l, d, eta, parameter)

        throughput_list.append(throughput)
        power_list.append(power)

    throughput_array = np.array(throughput_list).reshape((case_num, 1))
    power_array = np.array(power_list).reshape((case_num, 1))
    filepath = 'Learn2MAC' + '_D_' + str(D) + '_n2_' +  + '_idx_' + str(idx)
    np.save(filepath + 'throuput', throughput_array)
    np.save(filepath + 'power', power_array)