import numpy as np
from Convert import index2state, numpy2excel

def getTrans_prob(D, pb1, pb2, ps1, ps2, pt1):
    trans_prob_W = np.zeros(shape=(2**(2*D)*4, 2**(2*D)*4), dtype=float)
    # agent2 等待
    for i in range(trans_prob_W.shape[0]):
        for j in range(trans_prob_W.shape[1]):
            L1, L2, _ = index2state(index=i, D=D)
            L1_, L2_, O_ = index2state(index=j, D=D)
            # 因为agent2是等待，所以L2只是前移一位并且随机来包
            if O_ == 'S': # agent2 发送成功，不可能
                continue
            if L2_ == L2[1:] + [0]:     # agent2 没有新的数据包
                if O_ == 'B' and sum(L1) != 0:  # agent1 有数据包且发送成功
                    if L1[0] != 0:  # agent1 首位有数据包
                        if L1_ == L1[1:] + [0]: # agent1 前移一位并且没有新的数据包
                            trans_prob_W[i][j] = pt1 * ps1 * (1 - pb1) * (1 - pb2)      # agent1发送首位的数据包，成功，且没有新的数据包，同时agent2 没有新的数据包
                        if L1_ == L1[1:] + [1]: # agent1 前移一位并且有新的数据包
                            trans_prob_W[i][j] = pt1 * ps1 * pb1 * (1 - pb2)            # agent1发送首位的数据包，成功，且有新的数据包，同时agent2 没有新的数据包
                    if L1[0] == 0:           # agent1 有数据包且不在首位
                        L1[L1.index(1)] = 0 # agent1 发送出队列最前面的数据包
                        if L1_ == L1[1:] + [0]:  # agent1 前移一位并且没有新的数据包
                            trans_prob_W[i][j] = pt1 * ps1 * (1 - pb1) * (1 - pb2)      # agent1发送非首位的数据包，成功，且没有新的数据包，同时agent2 没有新的数据包
                        if L1_ == L1[1:] + [1]: # agent1 前移一位并且有新的数据包
                            trans_prob_W[i][j] = pt1 * ps1 * pb1 * (1 - pb2)            # agent1发送非首位的数据包，成功，且有新的数据包，同时agent2 没有新的数据包

                if O_ == 'I' and sum(L1) == 0:      # 信道空闲
                    if L1_ == L1[1:] + [0]:         # agent1 前移一位且没有新的数据包
                        trans_prob_W[i][j] = (1 - pb1) * (1 - pb2)  # agent1 没有新的数据包，agent2 没有新的数据包
                    if L1_ == L1[1:] + [1]:
                        trans_prob_W[i][j] = pb1 * (1 - pb2)        # agent1 有新的数据包，agent 没有新的数据包

                if O_ == 'I' and sum(L1) != 0:      # agent1有数据包，且agent1 和agent2 都不发送数据包
                    if L1_ == L1[1:] + [0]:         # agent1 前移一位且没有新的数据包
                        trans_prob_W[i][j] = (1 - pt1) * (1 - pb1) * (1 - pb2)  # agent1 没有新的数据包，agent2 没有新的数据包
                    if L1_ == L1[1:] + [1]:
                        trans_prob_W[i][j] = (1 - pt1) * pb1 * (1 - pb2)        # agent1 有新的数据包，agent 没有新的数据包

                if O_ == 'F' and sum(L1) != 0:      # agent1 有数据包且发送失败
                    if L1_ == L1[1:] + [0]:         # agent1 没有新的数据包
                        trans_prob_W[i][j] = pt1 * (1 - ps1) * (1 - pb1) * (1 - pb2)    # agent1 有数据包，发送失败，且没有新的数据包， agent2 没有新的数据包
                    if L1_ == L1[1:] + [1]:         # agent1 有新的数据包
                        trans_prob_W[i][j] = pt1 * (1 - ps1) * pb1 * (1 - pb2)          # agent1 有数据包，发送失败，且有新的数据包， agent2 没有新的数据包

            if L2_ == L2[1:] + [1]:     # agent2 有新的数据包
                if O_ == 'B' and sum(L1) != 0:  # agent1 有数据包且发送成功
                    if L1[0] != 0:  # agent1 首位有数据包
                        if L1_ == L1[1:] + [0]: # agent1 前移一位并且没有新的数据包
                            trans_prob_W[i][j] = pt1 * ps1 * (1 - pb1) * pb2    # agent1发送首位的数据包，成功，且没有新的数据包，同时agent2 有新的数据包
                        if L1_ == L1[1:] + [1]: # agent1 前移一位并且有新的数据包
                            trans_prob_W[i][j] = pt1 * ps1 * pb1 * pb2          # agent1发送首位的数据包，成功，且有新的数据包，同时agent2 有新的数据包
                    if L1[0] == 0:           # agent1 有数据包且不在首位
                        L1[L1.index(1)] = 0 # agent1 发送出队列最前面的数据包
                        if L1_ == L1[1:] + [0]:  # agent1 前移一位并且没有新的数据包
                            trans_prob_W[i][j] = pt1 * ps1 * (1 - pb1) * pb2    # agent1发送首位的数据包，成功，且没有新的数据包，同时agent2 有新的数据包
                        if L1_ == L1[1:] + [1]: # agent1 前移一位并且有新的数据包
                            trans_prob_W[i][j] = pt1 * ps1 * pb1 * pb2          # agent1发送首位的数据包，成功，且有新的数据包，同时agent2 有新的数据包

                if O_ == 'I' and sum(L1) == 0:      # 信道空闲
                    if L1_ == L1[1:] + [0]:         # agent1 前移一位且没有新的数据包
                        trans_prob_W[i][j] = (1 - pb1) * pb2  # agent1 没有新的数据包，agent2 有新的数据包
                    if L1_ == L1[1:] + [1]:
                        trans_prob_W[i][j] = pb1 * pb2        # agent1 有新的数据包，agent 有新的数据包

                if O_ == 'I' and sum(L1) != 0:      # agent1有数据包，且agent1 和agent2 都不发送数据包
                    if L1_ == L1[1:] + [0]:         # agent1 前移一位且没有新的数据包
                        trans_prob_W[i][j] = (1 - pt1) * (1 - pb1) * pb2  # agent1 没有新的数据包，agent2 有新的数据包
                    if L1_ == L1[1:] + [1]:
                        trans_prob_W[i][j] = (1 - pt1) * pb1 * pb2        # agent1 有新的数据包，agent 有新的数据包

                if O_ == 'F' and sum(L1) != 0:      # agent1 有数据包且发送失败
                    if L1_ == L1[1:] + [0]:         # agent1 没有新的数据包
                        trans_prob_W[i][j] = pt1 * (1 - ps1) * (1 - pb1) * pb2    # agent1 有数据包，发送失败，且没有新的数据包， agent2 有新的数据包
                    if L1_ == L1[1:] + [1]:         # agent1 有新的数据包
                        trans_prob_W[i][j] = pt1 * (1 - ps1) * pb1 * pb2          # agent1 有数据包，发送失败，且有新的数据包， agent2 有新的数据包


    trans_prob_T = np.zeros(shape=(2**(2*D)*4, 2**(2*D)*4), dtype=float)
    # agent2 发送
    for i in range(trans_prob_T.shape[0]):
        for j in range(trans_prob_T.shape[1]):
            L1, L2, _ = index2state(index=i, D=D)
            L1_, L2_, O_ = index2state(index=j, D=D)
            if O_ == 'B': # agent1 发送成功，不可能
                continue
            if O_ == 'I': # 信道空闲，不可能
                continue
            if O_ == 'S' and sum(L2) != 0:  # agent2 有数据包且发送成功
                if sum(L1) == 0:            # agent1 没有数据包
                    if L1_ == L1[1:] + [0]: # L1 前移一位且没有新的数据包
                        if L2[0] != 0:           # agent2 首位有数据包
                            if L2_ == L2[1:] + [0]:     # L2 前移一位且没有新的数据包
                                trans_prob_T[i][j] = ps2 * (1 - pb1) * (1 - pb2)
                            if L2_ == L2[1:] + [1]:     # L2 前移一位且有新的数据包
                                trans_prob_T[i][j] = ps2 * (1 - pb1) * pb2
                        if L2[0] == 0:                  # agent2 有数据包且不在首位
                            L2[L2.index(1)] = 0         # agent2 发送出队列最前面的数据包
                            if L2_ == L2[1:] + [0]:     # L2 前移一位且没有新的数据包
                                trans_prob_T[i][j] = ps2 * (1 - pb1) * (1 - pb2)
                            if L2_ == L2[1:] + [1]:     # L2 前移一位且有新的数据包
                                trans_prob_T[i][j] = ps2 * (1 - pb1) * pb2
                    if L1_ == L1[1:] + [1]: # L1 前移一位且有新的数据包
                        if L2[0] != 0:           # agent2 首位有数据包
                            if L2_ == L2[1:] + [0]:     # L2 前移一位且没有新的数据包
                                trans_prob_T[i][j] = ps2 * pb1 * (1 - pb2)
                            if L2_ == L2[1:] + [1]:     # L2 前移一位且有新的数据包
                                trans_prob_T[i][j] = ps2 * pb1 * pb2
                        if L2[0] == 0:                  # agent2 有数据包且不在首位
                            L2[L2.index(1)] = 0         # agent2 发送出队列最前面的数据包
                            if L2_ == L2[1:] + [0]:     # L2 前移一位且没有新的数据包
                                trans_prob_T[i][j] = ps2 * pb1 * (1 - pb2)
                            if L2_ == L2[1:] + [1]:     # L2 前移一位且有新的数据包
                                trans_prob_T[i][j] = ps2 * pb1 * pb2
                
                if sum(L1) != 0:        # agent1 有数据包
                    if L1_ == L1[1:] + [0]: # L1 前移一位且没有新的数据包
                        if L2[0] != 0:           # agent2 首位有数据包
                            if L2_ == L2[1:] + [0]:     # L2 前移一位且没有新的数据包
                                trans_prob_T[i][j] = ps2 * (1 - pb1) * (1 - pb2) * (1 - pt1)
                            if L2_ == L2[1:] + [1]:     # L2 前移一位且有新的数据包
                                trans_prob_T[i][j] = ps2 * (1 - pb1) * pb2 * (1 - pt1)
                        if L2[0] == 0:                  # agent2 有数据包且不在首位
                            L2[L2.index(1)] = 0         # agent2 发送出队列最前面的数据包
                            if L2_ == L2[1:] + [0]:     # L2 前移一位且没有新的数据包
                                trans_prob_T[i][j] = ps2 * (1 - pb1) * (1 - pb2) * (1 - pt1)
                            if L2_ == L2[1:] + [1]:     # L2 前移一位且有新的数据包
                                trans_prob_T[i][j] = ps2 * (1 - pb1) * pb2 * (1 - pt1)
                    if L1_ == L1[1:] + [1]: # L1 前移一位且有新的数据包
                        if L2[0] != 0:           # agent2 首位有数据包
                            if L2_ == L2[1:] + [0]:     # L2 前移一位且没有新的数据包
                                trans_prob_T[i][j] = ps2 * pb1 * (1 - pb2) * (1 - pt1)
                            if L2_ == L2[1:] + [1]:     # L2 前移一位且有新的数据包
                                trans_prob_T[i][j] = ps2 * pb1 * pb2 * (1 - pt1)
                        if L2[0] == 0:                  # agent2 有数据包且不在首位
                            L2[L2.index(1)] = 0         # agent2 发送出队列最前面的数据包
                            if L2_ == L2[1:] + [0]:     # L2 前移一位且没有新的数据包
                                trans_prob_T[i][j] = ps2 * pb1 * (1 - pb2) * (1 - pt1)
                            if L2_ == L2[1:] + [1]:     # L2 前移一位且有新的数据包
                                trans_prob_T[i][j] = ps2 * pb1 * pb2 * (1 - pt1)
            
            if O_ == 'F' and sum(L2) != 0: # 冲突、信道噪声
                if sum(L1) != 0: # agent1 有数据包
                    if L1_ == L1[1:] + [0]:     # agent1 前移一位且没有新的数据包
                        if L2_ == L2[1:] + [0]: # agent2 前移一位且没有新的数据包
                            trans_prob_T[i][j] = pt1 * (1 - pb1) * (1 - pb2) + (1 - pt1) * (1 - ps2) * (1 - pb1) * (1 - pb2)
                        if L2_ == L2[1:] + [1]: # agent2 前移一位且有新的数据包
                            trans_prob_T[i][j] = pt1 * (1 - pb1) * pb2 + (1 - pt1) * (1 - ps2) * (1 - pb1) * pb2
                    if L1_ == L1[1:] + [1]:     # agent1 前移一位且有新的数据包
                        if L2_ == L2[1:] + [0]: # agent2 前移一位且没有新的数据包
                            trans_prob_T[i][j] = pt1 * pb1 * (1 - pb2) + (1 - pt1) * (1 - ps2) * pb1 * (1 - pb2)
                        if L2_ == L2[1:] + [1]: # agent2 前移一位且有新的数据包
                            trans_prob_T[i][j] = pt1 * pb1 * pb2 + (1 - pt1) * (1 - ps2) * pb1 * pb2
                
                # 只有agent2 发送数据包且失败
                if sum(L1) == 0: # agent1 没有数据包
                    if L1_ == L1[1:] + [0]:     # agent1 前移一位且没有新的数据包
                        if L2_ == L2[1:] + [0]: # agent2 前移一位且没有新的数据包
                            trans_prob_T[i][j] = (1 - ps2) * (1 - pb1) * (1 - pb2)
                        if L2_ == L2[1:] + [1]: # agent2 前移一位且有新的数据包
                            trans_prob_T[i][j] = (1 - ps2) * (1 - pb1) * pb2
                    if L1_ == L1[1:] + [1]:     # agent1 前移一位且有新的数据包
                        if L2_ == L2[1:] + [0]: # agent2 前移一位且没有新的数据包
                            trans_prob_T[i][j] = (1 - ps2) * pb1 * (1 - pb2)
                        if L2_ == L2[1:] + [1]: # agent2 前移一位且有新的数据包
                            trans_prob_T[i][j] = (1 - ps2) * pb1 * pb2

    return trans_prob_W, trans_prob_T

def testTrans_prob():
    pb1 = 0.3
    pb2 = 0.4
    ps1 = 0.7
    ps2 = 0.6
    pt1 = 0.5

    # N = 2
    D = 2
    trans_prob_W, trans_prob_T = getTrans_prob(D=D, pb1=pb1, pb2=pb2, ps1=ps1, ps2=ps2, pt1=pt1)
    numpy2excel(trans_prob=trans_prob_W, filename='trans_prob_W')
    numpy2excel(trans_prob=trans_prob_T, filename='trans_prob_T')
    print('OK')

if __name__ == "__main__":
    testTrans_prob()




