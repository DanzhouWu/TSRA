import numpy as np
def merge_pattern(agents_list):
    D = agents_list[1].N # deadline
    N = len(agents_list) # users number
    patterns = [0] * D
    for i in range(D):
        for j in range(N):
            patterns[i] += agents_list[j].actions_list[i]
    return patterns

def adjust_R(pattern, pre_queue, others, pb, ps):
    D = len(pattern)
    for slot_idx in range(D):
        assert others[slot_idx] >= 0
        if pattern[slot_idx] == 1 and 1 in pre_queue \
            and others[slot_idx] == 0 and ps > np.random.uniform():
            return 1
        else:
            pre_queue = pre_queue[1:] + [1 if pb > np.random.uniform() else 0]
    return 0