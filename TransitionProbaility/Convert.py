'''
    '3' -> '0101'   (if length = 4)
    input  ：
        dec_num : type = string;
        length  : type = int;
    output :
        out     : type = string.
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
    D = 2,  index = '27' -> L1 = [0, 1], L2 = [1, 0], O = 'C'
    input:  type(index) = int, type(D) = int
    outpt:  type(L1) = list, type(L1) = list, type(D) = int
'''
def index2state(index, D):
    if index >= 2**(2*D)*4 or index < 0 or type(index) != int or type(D) != int:
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
        l_bin = dec2bin(str(l_index), 2*D)
        L1 = str2list(l_bin[:D])
        L2 = str2list(l_bin[-D:])
        return L1, L2, O
    except IOError:
        print('input error!')

'''
    D = 2, L1 = [0, 1], L2 = [1, 0], O = 'C' -> index = 27
    input   : type(L1) = type(L2) = list, type(O) = str, type(D) = int
    output  : type(index) = int
'''
def state2index(L1, L2, O, D):
    if type(D) != int or type(L1) != list or type(L2) != list or type(O) != str:
        raise  EOFError('parameter type is wrong!')
    if len(L1) != D or len(L2) != D:
        raise EOFError('There are some error between L1, L2 and D')
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
def numpy2excel(trans_prob, filename):
    data = pd.DataFrame(trans_prob)
    writer = pd.ExcelWriter(filename + '.xlsx')
    data.to_excel(writer, sheet_name='Sheet1', float_format='%.5f')
    writer.save()
    writer.close()



def testCovert():
    index = 60
    L1, L2, O = index2state(index=index, D=3)
    print(L1)
    print(L2)
    print(O)
    index = state2index(L1, L2, O, D=3)
    print(index)

if __name__ == "__main__":
    testCovert()



