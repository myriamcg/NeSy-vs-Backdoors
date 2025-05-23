
#  NeSy_vs_Backdoors research project

This file explains the structure of the NeSy_vs_Backdoors repository, the experiments that were done so far, overall idea of results.

The project focuses on the robustness of a Logic Tensor Network (LTN) against BadNet attacks. The project follows 3 milestones: building an LTN model on a few representative tasks, building a BadNet attack on a Neural Network, and then running the attack on the LTN model, on at least one representative task.

## Building the LTN model

## Setup

In order to make the LTN examples run, first you need to create a virtual environment in the [logictensornetwork](logictensornetwork) folder. You can do this by running the following command:

```bash
python -m venv .venv
```

Then, you need to activate the virtual environment. You can do this by running the following command:

```bash
source ltn_env/bin/activate
```

Then, you need to install the requirements. You can do this by running the following command:

```bash
pip install -r requirements.txt
```

Then, in the base folder, you will use the interpreter from the virtual environment from the logictensornetwork folder. You can do this by manually setting the interpreter to be the one from the environment of the logictensornetworks ( can be found at `logictensornetwork/.venv/Scripts/python.exe`). 
The python version is Python 3.12, pip version is 23.2.1. By doing this setup, you can ensure that you won't encounter deprecated packages errors. 



## Experiments

For the first milestone, the https://github.com/logictensornetworks/logictensornetworks repository was used as a reference. This repository contains a tutorial on how to use the LTN framework and a few examples on representative tasks for LTN https://github.com/logictensornetworks/logictensornetworks/tree/master/examples. From these examples, I chose using the **mnist** and the **smokes_friends_cancer** problems. 

Although the used repository has the latest ltn version, the examples weren't updated to run on it, so in this repository I first had to modify the ipynb files from those directories to make them runnable. The experiments that were run can be found in the [mnist directory](logictensornetwork/examples/mnist) and [smokes_friends_cancer](logictensornetwork%2Fexamples%2Fsmokes_friends_cancer). After making the code runnable, I focused more on the **mnist** problem, by running both the single_digit_additio and multi_digit_addition notebooks. Then, I  changed the operation that the model was being trained on (from addition to subtraction), experiment which can be found in [[single_digits_difference.ipynb](logictensornetwork%2Fexamples%2Fmnist%2Fsingle_digits_difference.ipynb)] .and then I tried adding more symbolic knowledge to the model, experiment which can be found in [[single_digits_addition_more_symbolic_knowledge.ipynb](logictensornetwork%2Fexamples%2Fmnist%2Fsingle_digits_addition_more_symbolic_knowledge.ipynb)].

The first experiment was a simple one, where I trained the model to learn the subtraction of two digits. The second experiment was more complex, where I added more symbolic knowledge to the model, by adding a few rules that the model should follow. The results of these experiments can be found in the respective ipynb files.

[all_experiments_mnist.ipynb](logictensornetwork%2Fexamples%2Fmnist%2Fall_experiments_mnist.ipynb) contains the more important experiments on MNIST, where I also played around with p_schedules, learning rates, number of epochs and layers.
That file also contains the observations made on the results of the experiments and a final overview of how LTN acts with more or less symbolic knowledge.
## Building the BadNet attack

## Running the BadNet on the LTN



