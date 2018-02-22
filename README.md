# Addressing Function Approximation Error in Actor-Critic Methods
Scott Fujimoto, Herke van Hoof and David Meger

PyTorch implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3). If you use our code or data please cite the paper: []

Method is tested on [MuJoCo](http://www.mujoco.org/) continuous control tasks in [OpenAI gym](https://github.com/openai/gym). 
Networks are trained using [PyTorch](https://github.com/pytorch/pytorch). 

The paper results can be reproduced exactly running the experiments.sh script. 
Experiments on single environments can be run by calling
```
python2 main.py --env HalfCheetah-v1
```

Hyper-parameters can be modified with different arguments to main.py. We include an implementation of DDPG to compare differences with TD3, this is not the implementation of DDPG used in the paper. 

Algorithms which TD3 compares against (PPO, TRPO, ACKTR, DDPG) can be found at [OpenAI baselines repository](https://github.com/openai/baselines). 

Learning curves found in the paper are found under /learning_curves. Each learning curve are formatted as NumPy arrays of 201 evaluations (201,), where each evaluation corresponds to the average total reward from running the policy for 10 episodes with no exploration. The first evaluation is the randomly initialized policy network (unused in the paper). 