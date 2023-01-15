# Self-Driving Car with Deep Learning, Reinforcement Learning - MSc Thesis Work
## Abstract

The prevailing R&D trends of the recent years in the automotive industry have
been e-mobility, autonomous vehicles and intelligent driving assistant systems. These
architectures are not only expected to make driving easier and a more comfortable
experience, but also to decrease the number of accidents on the roads. The term intelligent
systems usually covers solutions, which use different soft-computing techniques (e.g.:
*fuzzy logic*, *gradient-based optimization*, *machine and deep learning*). Among these, deep
learning and reinforcement learning are currently the most widely researched fields.
However, these are both computation-heavy techniques that require a tremendous amount
of high quality data and a high fidelity simulation environment to conduct a proper
training. But even then, it is still not possible to cover all situations that can occur in real-life traffic. Another difficulty is – the so-called – '*freezing robot*' problem. This term
describes the phenomenon, when the self-driving agent turns out to be so risk-averse that
in order to avoid collision under every circumstances, it becomes unable to carry out a
successful merge on the highway, or make a left turn at an intersection, as it can not
distinguish between situations when merging is considered safe and when it is not.

In my Master Thesis first, I implement a social-attention architecture with the DQN
*reinforcement learning* algorithm based on the *Social Attention for Autonomous Decision-Making in Dense Traffic*
article from *Edouard Leurent* and *Jean Mercat*. With
this architecture, the self-driving agent is able to focus, '*pay attention*' to the behaviour
of other vehicles in traffic and weight their importance in terms of the current situation.
This structure is tested in a simulation, where the self-driving vehicle has to make a left
turn at an intersection with dense traffic. Then, I modify the architecture based on my
own idea, to use the self-implemented PPO algorithm for training. In my work, the <Python>
programming language with the <PyTorch> *deep learning* module is used for implementing
the attention-based architecture and the reinforcement learning algorithms. As simulation
environment, the <highway-env> is selected. This is a simple training environment (or
'*gym*') library which is capable of simulating a wide variety of different traffic situations
while requiring relatively low computational capacity compared to other, more complex
environments.
