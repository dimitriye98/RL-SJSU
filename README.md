RL@SJSU
=======
This is a demo repository for RL techniques being discussed at ML@SJSU club at 
San Jose State University

Q-Learning
----------
Run the **Q-Learning Agent** section in the 
[qlearning notebook](qlearning.ipynb) to see a demo of basic Q-Learning with no 
replay buffer learning to land in the lunar lander environment.

![](pics/SimpleDQN_Lander.gif)

`ql.py` contains a minimal online q-learning algorithm for tabular settings.

Deep Q-Learning
---------------
`dql.py` adds a simple modification to the ql.py, making it capable 
(somewhat) to learn in continuous state spaces.