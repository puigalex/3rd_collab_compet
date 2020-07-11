[//]: # (Image References)
[Scores]: ./resources/Score_done.png 
[num_scores]: ./resources/avg_scores.png


# Report for p3_collab_compet from Deep Reinforcement Learning Nano Degree

## Learning Algorithm 

### DDPG

Deep Deterministic Policy Gradient is an algorithm in reinforcement learning that combines two approaches for deep reinforcement learning, Q-Learning and Policy based methods. As we learned from the previous lessons, Q-Learning is a good method to use in specific cases to teach agents, however if we have a continuous actions/state space this can become very complicated, of course we can try to discretise the problem, other solution is to use Policy based methods, in which we are able to estimate the optimal policy without the need to estimate the optimal value function. But in this case with DDPG we'll use an actor-critic agent in which we merge this two concepts in order to have a better balnce between our variance and bias.

This architecture consists of two main parts, the actor and the critic.

The critic is a NN that is in charge of evaluating the value function with the best action identified by the actor, then we calculate the advantage funcion to train the other NN that is the actor.

### MADDPG

Now, in many situations we need to train on environments where mutiple agents interact with each other, on DDPG we are training an agent that interact with and environment (Interactions can be **coordination, competition, negotiation or communication** between agents), but if we want to train several agents that interact between them we need a new approach, this approach is calles Multi Agent DDPG.

In MADDPG the critic of each agent has information of all the environment, including other agents but the actor can only have visibility of its corresponding agent.

The first agent that solved the task had the following architecture

- Actor
    - Input: (24 neurons, from the state space)
    - Hidden: (512, 256 neurons)
    - Output: 2 Neurons (Action space)
- Critic
    - Input: (48 neurons, because of two agents with 24 parameters each)
    - Hidden: (516 [512 + 4], 256 neurons)
    - Output: 1 Neuron for action vector



## Findings (Human Learning)

Frankly I thought it was going to be an easier task, just make a copuple adaptations to DDPG project and done, however I had a couple of issues with one of my agents getting "stucked" during training to later realice I was updating the actor incorrectly.

I ran several exercises in my local machine and saw that increasing the buffer size helped stabilizing the learning, however it also depleted my GPU memory (GTX 1050). I also want to start playing more with the noise in order to see if I get better results.

I got to the required average score in 2400 episodes, at this moment each episode took longer and longer to run, so I just let is run for a couple hundred more episodes to see it held the average above 0.5 but with time or maybe working on a VM I'd like to let it run for at least 10K episodes and see what happens.

I rally like this project, and would like to try the soccer environment, however I fist need to create a really robust model for this problem.

This is the graph showing the average score over the last 100 episodes </br>
![][Scores]
</br>
![][num_scores]




## Ideas for future work 

I've been trying new NN architectures in order to see if I can stabilice the agent's learning. such as doing some dropout or batch normalization between some layers of the model. At the moment I haven't got any substantial new findings but will keep experimenting to see how can I create agents that are able to learn faster and more stable
