import random, argparse, copy
import numpy as np
from tqdm import tqdm
from DQN_agent import DQN_AGENT
from environment import ENVIRONMENT
from until import return_next_state
random.seed(20)
np.random.seed(1)

n_nodes = 2 # number of nodes
n_actions = 2 # number of actions

M = 20 # state length
E = 1000 # memory size
F = 20 # target network update frequency
B = 64 # mini-batch size
gamma = 0.9 # discount factor

alpha = 1 # fairness index

max_iter = int(5e4)
idx = 1

def main(n2, D, parameter, iteration=int(1e5)):
    agent_list = [DQN_AGENT(D=D, 
                            arrival_rate=parameter[i], 
                            state_size=int(8*M),
                            n_actions=2, 
                            n_nodes=2,
                            memory_size=E,
                            replace_target_iter=F,
                            batch_size=B,
                            ) for i in range(n2)]

    env = ENVIRONMENT(channels=parameter[n2:], agent_list=agent_list)

    reward_list = []
    energy_list = []

    state = [[0] * int(8*M) for _ in range(n2)]
    next_state = [[0] * int(8*M) for _ in range(n2)]
    for t in tqdm(range(iteration)):
        for i in range(n2):
            env.agent_list[i].choose_action(np.array(state[i]))

        reward, energy, observations = env.step(time=t) 

        reward_list.append(reward)
        energy_list.append(energy)

        for i in range(n2): 
            env.agent_list[i].update_queue(observation=observations[i])
            next_state[i], agent_reward, others_reward = return_next_state(i, state[i], env.agent_list, observations, reward)
            env.agent_list[i].store_transition(state[i], env.agent_list[i].action, agent_reward, others_reward, next_state[i])

        if t > 100 and t % 5 == 0:
            for i in range(n2):
                env.agent_list[i].learn() 

        state = copy.deepcopy(next_state)

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

    case_num = 20

    parameters = np.load('../RandomParameters_seed_10_100.npy')
    
    throughput_list = []
    power_list = []
    for parameter in parameters[case_num*idx: case_num*(idx+1)]:

        throughput, power = main(n2, D, parameter)

        throughput_list.append(throughput)
        power_list.append(power)

    throughput_array = np.array(throughput_list).reshape((case_num, 1))
    power_array = np.array(power_list).reshape((case_num, 1))
    filepath = 'DLMA' + 'D_' + str(D) + '_n2_' + str(n2) + '_idx_' + str(idx)
    np.save(filepath + 'throuput', throughput_array)
    np.save(filepath + 'power', power_array)
