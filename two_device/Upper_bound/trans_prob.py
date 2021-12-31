import numpy as np
from until import index2state, numpy2excel

# get transition probability matrix
def getTrans_prob(D, D_, pb1, pb2, ps1, ps2, pt1):
    # Transition probability matrix for "WAIT"
    trans_prob_W = np.zeros(shape=(2**(D_+D)*4, 2**(D_+D)*4), dtype=float)

    for i in range(trans_prob_W.shape[0]):
        for j in range(trans_prob_W.shape[1]):
            L1, L2, _ = index2state(index=i, D=D, D_=D_)
            L1_, L2_, O_ = index2state(index=j, D=D, D_=D_)

            if O_ == 'S': 
                continue
            if L2_ == L2[1:] + [0]:    
                if O_ == 'B' and sum(L1) != 0: 
                    if L1[0] != 0:  
                        if L1_ == L1[1:] + [0]: 
                            trans_prob_W[i][j] = pt1 * ps1 * (1 - pb1) * (1 - pb2)      
                        if L1_ == L1[1:] + [1]: 
                            trans_prob_W[i][j] = pt1 * ps1 * pb1 * (1 - pb2)            
                    if L1[0] == 0:           
                        L1[L1.index(1)] = 0 
                        if L1_ == L1[1:] + [0]:  
                            trans_prob_W[i][j] = pt1 * ps1 * (1 - pb1) * (1 - pb2)      
                        if L1_ == L1[1:] + [1]: 
                            trans_prob_W[i][j] = pt1 * ps1 * pb1 * (1 - pb2)            

                if O_ == 'I' and sum(L1) == 0:      
                    if L1_ == L1[1:] + [0]:         
                        trans_prob_W[i][j] = (1 - pb1) * (1 - pb2)  
                    if L1_ == L1[1:] + [1]:
                        trans_prob_W[i][j] = pb1 * (1 - pb2)        

                if O_ == 'I' and sum(L1) != 0:      
                    if L1_ == L1[1:] + [0]:         
                        trans_prob_W[i][j] = (1 - pt1) * (1 - pb1) * (1 - pb2)  
                    if L1_ == L1[1:] + [1]:
                        trans_prob_W[i][j] = (1 - pt1) * pb1 * (1 - pb2)        

                if O_ == 'F' and sum(L1) != 0:      
                    if L1_ == L1[1:] + [0]:         
                        trans_prob_W[i][j] = pt1 * (1 - ps1) * (1 - pb1) * (1 - pb2)    
                    if L1_ == L1[1:] + [1]:         
                        trans_prob_W[i][j] = pt1 * (1 - ps1) * pb1 * (1 - pb2)          

            if L2_ == L2[1:] + [1]:     
                if O_ == 'B' and sum(L1) != 0:  
                    if L1[0] != 0:  
                        if L1_ == L1[1:] + [0]: 
                            trans_prob_W[i][j] = pt1 * ps1 * (1 - pb1) * pb2    
                        if L1_ == L1[1:] + [1]: 
                            trans_prob_W[i][j] = pt1 * ps1 * pb1 * pb2          
                    if L1[0] == 0:           
                        L1[L1.index(1)] = 0 
                        if L1_ == L1[1:] + [0]: 
                            trans_prob_W[i][j] = pt1 * ps1 * (1 - pb1) * pb2    
                        if L1_ == L1[1:] + [1]: 
                            trans_prob_W[i][j] = pt1 * ps1 * pb1 * pb2          

                if O_ == 'I' and sum(L1) == 0:      
                    if L1_ == L1[1:] + [0]:         
                        trans_prob_W[i][j] = (1 - pb1) * pb2  
                    if L1_ == L1[1:] + [1]:
                        trans_prob_W[i][j] = pb1 * pb2        

                if O_ == 'I' and sum(L1) != 0:      
                    if L1_ == L1[1:] + [0]:         
                        trans_prob_W[i][j] = (1 - pt1) * (1 - pb1) * pb2  
                    if L1_ == L1[1:] + [1]:
                        trans_prob_W[i][j] = (1 - pt1) * pb1 * pb2        

                if O_ == 'F' and sum(L1) != 0:      
                    if L1_ == L1[1:] + [0]:         
                        trans_prob_W[i][j] = pt1 * (1 - ps1) * (1 - pb1) * pb2    
                    if L1_ == L1[1:] + [1]:         
                        trans_prob_W[i][j] = pt1 * (1 - ps1) * pb1 * pb2          

    trans_prob_T = np.zeros(shape=(2**(D_+D)*4, 2**(D_+D)*4), dtype=float)
    # Transition probability matrix for "TRANSMIT"
    for i in range(trans_prob_T.shape[0]):
        for j in range(trans_prob_T.shape[1]):
            L1, L2, _ = index2state(index=i, D=D, D_=D_)
            L1_, L2_, O_ = index2state(index=j, D=D, D_=D_)
            if O_ == 'B': 
                continue
            if O_ == 'I': 
                continue
            if O_ == 'S' and sum(L2) != 0:  
                if sum(L1) == 0:            
                    if L1_ == L1[1:] + [0]: 
                        if L2[0] != 0:      
                            if L2_ == L2[1:] + [0]:     
                                trans_prob_T[i][j] = ps2 * (1 - pb1) * (1 - pb2)
                            if L2_ == L2[1:] + [1]:     
                                trans_prob_T[i][j] = ps2 * (1 - pb1) * pb2
                        if L2[0] == 0:                  
                            L2[L2.index(1)] = 0         
                            if L2_ == L2[1:] + [0]:     
                                trans_prob_T[i][j] = ps2 * (1 - pb1) * (1 - pb2)
                            if L2_ == L2[1:] + [1]:     
                                trans_prob_T[i][j] = ps2 * (1 - pb1) * pb2
                    if L1_ == L1[1:] + [1]: 
                        if L2[0] != 0:      
                            if L2_ == L2[1:] + [0]:     
                                trans_prob_T[i][j] = ps2 * pb1 * (1 - pb2)
                            if L2_ == L2[1:] + [1]:     
                                trans_prob_T[i][j] = ps2 * pb1 * pb2
                        if L2[0] == 0:                  
                            L2[L2.index(1)] = 0         
                            if L2_ == L2[1:] + [0]:     
                                trans_prob_T[i][j] = ps2 * pb1 * (1 - pb2)
                            if L2_ == L2[1:] + [1]:     
                                trans_prob_T[i][j] = ps2 * pb1 * pb2

                if sum(L1) != 0:        
                    if L1_ == L1[1:] + [0]: 
                        if L2[0] != 0:       
                            if L2_ == L2[1:] + [0]:    
                                trans_prob_T[i][j] = ps2 * (1 - pb1) * (1 - pb2) * (1 - pt1)
                            if L2_ == L2[1:] + [1]:    
                                trans_prob_T[i][j] = ps2 * (1 - pb1) * pb2 * (1 - pt1)
                        if L2[0] == 0:                 
                            L2[L2.index(1)] = 0        
                            if L2_ == L2[1:] + [0]:    
                                trans_prob_T[i][j] = ps2 * (1 - pb1) * (1 - pb2) * (1 - pt1)
                            if L2_ == L2[1:] + [1]:    
                                trans_prob_T[i][j] = ps2 * (1 - pb1) * pb2 * (1 - pt1)
                    if L1_ == L1[1:] + [1]: 
                        if L2[0] != 0:          
                            if L2_ == L2[1:] + [0]:    
                                trans_prob_T[i][j] = ps2 * pb1 * (1 - pb2) * (1 - pt1)
                            if L2_ == L2[1:] + [1]:    
                                trans_prob_T[i][j] = ps2 * pb1 * pb2 * (1 - pt1)
                        if L2[0] == 0:                 
                            L2[L2.index(1)] = 0        
                            if L2_ == L2[1:] + [0]:    
                                trans_prob_T[i][j] = ps2 * pb1 * (1 - pb2) * (1 - pt1)
                            if L2_ == L2[1:] + [1]:    
                                trans_prob_T[i][j] = ps2 * pb1 * pb2 * (1 - pt1)

            if O_ == 'F' and sum(L2) != 0: 
                if sum(L1) != 0: 
                    if L1_ == L1[1:] + [0]:   
                        if L2_ == L2[1:] + [0]: 
                            trans_prob_T[i][j] = pt1 * (1 - pb1) * (1 - pb2) + (1 - pt1) * (1 - ps2) * (1 - pb1) * (1 - pb2)
                        if L2_ == L2[1:] + [1]: 
                            trans_prob_T[i][j] = pt1 * (1 - pb1) * pb2 + (1 - pt1) * (1 - ps2) * (1 - pb1) * pb2
                    if L1_ == L1[1:] + [1]:     
                        if L2_ == L2[1:] + [0]: 
                            trans_prob_T[i][j] = pt1 * pb1 * (1 - pb2) + (1 - pt1) * (1 - ps2) * pb1 * (1 - pb2)
                        if L2_ == L2[1:] + [1]: 
                            trans_prob_T[i][j] = pt1 * pb1 * pb2 + (1 - pt1) * (1 - ps2) * pb1 * pb2
                
                
                if sum(L1) == 0:  
                    if L1_ == L1[1:] + [0]:      
                        if L2_ == L2[1:] + [0]: 
                            trans_prob_T[i][j] = (1 - ps2) * (1 - pb1) * (1 - pb2)
                        if L2_ == L2[1:] + [1]: 
                            trans_prob_T[i][j] = (1 - ps2) * (1 - pb1) * pb2
                    if L1_ == L1[1:] + [1]:     
                        if L2_ == L2[1:] + [0]: 
                            trans_prob_T[i][j] = (1 - ps2) * pb1 * (1 - pb2)
                        if L2_ == L2[1:] + [1]: 
                            trans_prob_T[i][j] = (1 - ps2) * pb1 * pb2

    return trans_prob_W, trans_prob_T

if __name__ == "__main__":
    pb1 = 0.5
    ps1 = 0.7
    pt1 = 0.4

    pb2 = 0.4
    ps2 = 0.6

    D = 2
    D_= 2

    trans_prob_W, trans_prob_T = getTrans_prob(D=D, D_=D_, pb1=pb1, pb2=pb2, ps1=ps1, ps2=ps2, pt1=pt1)
    numpy2excel(np_data=trans_prob_W, filename='trans_prob_W')
    numpy2excel(np_data=trans_prob_T, filename='trans_prob_T')
    print('OK')





