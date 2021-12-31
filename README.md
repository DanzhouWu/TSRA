# Introduction
This project is the code for our works:
 - **Reinforcement Learning for Improved Random   Access in Delay-Constrained Heterogeneous  Industrial IoT Networks**
 - **Reinforcement Learning Random Access for Delay-Constrained Heterogeneous Wireless Networks: A Two-User Case**

Note that The code author of [DLMA](https://github.com/YidingYu/DLMA) is YidingYu.  We have made some adjustments for delay-constrained communication.
# Run
## Requirement
**If you want to run all algorithms in this project, you need to install some packets as follow:**
- python = 3.6  
- tqdm  
- psutil
- numpy
- tensorflow-gpu = 1.14  
- keras = 2.3

## Two-device case

The code of two-device case is in the folder "two_device".

You can run the algorithm by entering the corresponding folder and using the following command:
``
python main.py
``

And there is an example for how to calculate the transition probability matrix for "Upper Bound" in the folder "TransitionProbaility".

## Multi-device case

The code of two-device case is in the folder “multi_device”.

You can run the algorithm by entering the corresponding folder and using the following command:

``
python main.py
``

There are some arguments you can change for different settings.

For example:

``
python main.py -D 10 -n1 3 -n2 100
``

means that the deadline $D$ is 10, the  number of uncontrollable devices $N_1$ is 3 and the number of TSRA $N_2$ is 100.

# Citation
If you find this project useful, we would be grateful if you cite our paper.

Our paper for two-device problem has been accepted by [2021 GLOBECOM Workshop on Towards Native-AI Wireless Networks.](https://globecom2021.ieee-globecom.org/workshop/ws-16-workshop-towards-native-ai-wireless-networks/program)

