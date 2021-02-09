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




'''
    '3' -> '0101'   (if length = 4)
    input  ：
        dec_num : type = str;
        length  : type = int;
    output :
        out     : type = str.
'''
def dec2bin(dec_num, length):
    if type(dec_num) != str or type(length) != int:
        raise  EOFError('parameter type is wrong!')
    try:
        out = ''
        while True:
            if int(dec_num) < 2:
                out += str(dec_num)
                break
            else:
                x = int(dec_num) % 2
                dec_num = int(dec_num) // 2
                out += str(x)
        
        if len(out) < length:
            out += '0' * (length - len(out))
        temp = list(out)
        temp.reverse()
        out = ''.join(temp)
        return out
    except IOError:
        print('input error!')

'''
    '101' -> '3'
    input   : bin_num :type = str
    output  : type = str
'''

def bin2dec(bin_num):
    return str(int(bin_num, 2))

'''
    [1,0,1] -> '101'
    input   : list_   : type = list
    output  : out     : type = str
'''
def list2str(list_):
    try:
        out = ''
        for i in list_:
            out += str(i)
        return out
    except IOError:
        print('input error!')

'''
    '101' -> [1, 0, 1]
    input : type(str_) = str
    output: type = list
'''
def str2list(str_):
    return [int(x) for x in str_]

'''
    index = 55 , D = 2, D_=3-> L1 = [0, 1], L2 = [1, 0, 1], O = 'F'
    input:  type(index) = int, type(D) = int, type(D_) = int
    outpt:  type(L1) = list, type(L1) = list, type(O) = str
'''
def index2state(index, D, D_):
    if index >= 2**(D+D_)*4 or index < 0 or type(index) != int or type(D) != int or type(D_) != int:
        raise  EOFError('parameter type is wrong!')
    try:
        o_index = index % 4
        if o_index == 0:
            O = 'B'
        if o_index == 1:
            O = 'S'
        if o_index == 2:
            O = 'I'
        if o_index == 3:
            O = 'F'
        l_index = index // 4
        l_bin = dec2bin(str(l_index), D+D_)
        L1 = str2list(l_bin[:D])
        L2 = str2list(l_bin[-D_:])
        return L1, L2, O
    except IOError:
        print('input error!')

'''
    L1 = [0, 1], L2 = [1, 0, 1], O = 'F' -> index = 55
    input   : type(L1) = type(L2) = list, type(O) = str, type(D) = int, type(D_) = int
    output  : type(index) = int
'''
def state2index(L1, L2, O):
    if type(L1) != list or type(L2) != list or type(O) != str:
        raise  EOFError('parameter type is wrong!')
    try:
        if O == 'B':
            o_index = 0
        if O == 'S':
            o_index = 1
        if O == 'I':
            o_index = 2
        if O == 'F':
            o_index = 3
        l1_str = list2str(L1)
        l2_str = list2str(L2)
        l_str = l1_str + l2_str
        l_index = int(bin2dec(l_str)) * 4
        index = int(l_index) + o_index
        return index
    except IOError:
        print('input error!')

def state2index_(L2, O, L1=[0]):
    if type(L1) != list or type(L2) != list or type(O) != str:
        raise  EOFError('parameter type is wrong!')
    try:
        if O == 'B':
            o_index = 0
        if O == 'S':
            o_index = 1
        if O == 'I':
            o_index = 2
        if O == 'F':
            o_index = 3
        l1_str = list2str(L1)
        l2_str = list2str(L2)
        l_str = l1_str + l2_str
        l_index = int(bin2dec(l_str)) * 4
        index = int(l_index) + o_index
        return index
    except IOError:
        print('input error!')

import pandas as pd
def numpy2excel(np_data, filename):
    data = pd.DataFrame(np_data)
    writer = pd.ExcelWriter(filename + '.xlsx')
    data.to_excel(writer, sheet_name='Sheet1', float_format='%.5f')
    writer.save()
    writer.close()

def testCovert():
    # index = 60
    # L1, L2, O = index2state(index=index, D=3, D_=2)
    # print(L1)
    # print(L2)
    # print(O)
    L1 = [1]
    L2 = [1]
    O = 'F'
    index = state2index(L1, L2, O)
    print(index)

if __name__ == "__main__":
    testCovert()


