[//]: # (Image References)

[env]: images/env.png "Environment"

# Project 3: Collaborative comept

# Introduction 

This is my submission for the 3rd project for the Udacity deep reinforcement learning nano degree, in this project I trained two agents to solve the table tennis environent using multi agent ddpg.

# Evironment spec

The environment consists of two agents (rackets) playing table tennis with a single ball every time one of the agents hits a ball it obtains a reward of 0.1 but if the ball goes out of bound or touches the table the agent obtains a reward of -0.1

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

# Success Criteria

In order to consider that the agents have learned they must obtain an average score of 0.5 over 100 consecutive episoder. The score is taking the maximum score of each episode (we have two scores per episode, one per agent)

# Requirements 

In order to run this project the following packages and envirnments should be installed in the machine. To do so, the following steps should be followed:

1. You can download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
2. Place the file in the `root/` folder of the project.

3. Install the requirements in the requirements.txt file


# Files

* **train.ipynb** This notebook contains the code to execute the training, initialice Unity environment, create the agents and measure the success criteria
* **maddpg.py** This file contains the buffer class to later sample from to train, the ddpg class (to initialice the agents) and the maddpg class to manage the training for the actors and critics from de ddpg class.
* **ddpg.py** This file contains the structure from the ddpg agents that maddpg calls upon, also some complementary classes. This file is similar to the one used in the p2 from this nano degree
* **model.py**  This file has the architecture for the neural network for the actor and the critis
