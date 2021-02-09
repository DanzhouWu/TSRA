# TSRA
The simulation code of TSRA

# Declaration
The code author of DLMA is YidingYu.  
https://github.com/YidingYu/DLMA  
We have made some adjustments to suit delay-constrained communication.  



## Requirement  
python = 3.6  
tqdm  
psutil  
tensorflow-gpu = 1.14  
keras = 2.3

## Run upper bound
```
cd Upper_bound  
python main.py
```  

## Run FSRA bound
```
cd FSRA  
python main.py
```  

## Run FSQA bound
```
cd FSQA  
python main.py
```  

## Run FSTA bound
```
cd FSTA  
python main.py
```  


## Run DLMA bound
```
cd DLMA
cd FNN
python main.py
```  
### or
```
cd DLMA
cd RNN
python main.py
```  
