from cvxopt import matrix, solvers
from cvxopt.modeling import variable, op
import numpy as np
import copy
from trans_prob import getTrans_prob

def unichainLP(D, D_, pb1, pt1, ps1, pb2, ps2):
    state_num = 2**(D+D_)*4
    action_num = 2

    x = variable(state_num*action_num, 'a')
    con1 = (x >= 0)
    con2 = (x <= 1)
    con3 = (sum(x) == 1)

    tp_w, tp_t = getTrans_prob(D, D_, pb1, pb2, ps1, ps2, pt1)

    tp_w = matrix(tp_w)
    tp_t = matrix(tp_t)
    con = []

    for j in range(state_num):
        item2 = 0
        for a in range(action_num):
            for s in range(state_num):
                if a == 0:
                    item2 += tp_w[s, j] * x[s]
                else:
                    item2 += tp_t[s, j] * x[s + state_num]
        item1 = x[j] + x[j+state_num]
        con.append((item1 - item2) == 0)
    con += [con1, con2, con3]

    reward_wait = np.zeros(shape=(state_num, 1))
    for i in range(state_num):
        if i % 4 == 0:
            reward_wait[i] = -1      # 'B'
            reward_wait[i+1] = -1    # 'S'
    reward_transmit = copy.deepcopy(reward_wait)

    r = np.concatenate((reward_wait, reward_transmit), axis=0).reshape(1,-1)
    r_m = matrix(r)
    lp1 = op(r_m*x, con)
    lp1.solve()

    x_ = np.array((x.value))

    opt_x = np.concatenate((x_[:state_num], x_[state_num:]), axis=1)

    policy = np.zeros(shape=(state_num, action_num), dtype=float)
    for s in range(state_num):
        m = 0
        for a in range(action_num):
            m += opt_x[s, a]

        if m != 0:
            for a in range(action_num):
                policy[s, a] = opt_x[s, a] / m
    return policy

if __name__ == '__main__':
    pb1 = 0.5
    ps1 = 0.7
    pt1 = 0.4

    pb2 = 0.4
    ps2 = 0.6

    D = 2
    D_= 2

    policy = unichainLP(D=D, D_=D_, pb1=pb1, pt1=pt1, ps1=ps1, pb2=pb2, ps2=ps2)
    print('ok')