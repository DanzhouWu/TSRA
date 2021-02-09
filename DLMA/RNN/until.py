def return_throughput(rewards, N = int(1e3)):
    temp_sum = 0
    throughput = []
    for i in range(len(rewards)):
        if i < N:
            temp_sum += rewards[i]
            throughput.append(temp_sum / (i+1))
        else:
            temp_sum += rewards[i] - rewards[i-N]
            throughput.append(temp_sum / N)
    return throughput

def return_queue_index(queue):
    queue_index = 0
    for i in queue:
        queue_index = 2 * queue_index + i
    return queue_index


# return channel observation
def return_observation(observation):
    if observation == 'S': # our agent transmits successfully
        return [1, 0, 0, 0]
    if observation == 'B': # other agent transmits successfully
        return [0, 1, 0, 0]
    if observation == 'I': # channel is idel
        return [0, 0, 1, 0]
    if observation == 'F': # collision or channel error
        return [0, 0, 0, 1]

def return_action(action, n_actions=2):
    one_hot_vector = [0] * n_actions
    one_hot_vector[action] = 1
    return one_hot_vector