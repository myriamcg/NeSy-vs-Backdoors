
#  NeSy_vs_Backdoors research project

This file explains the structure of the NeSy_vs_Backdoors repository, the experiments that were done so far, overall idea of results.

The project focuses on the robustness of a Logic Tensor Network (LTN) against BadNet attacks. The project follows 3 milestones: building an LTN model on a few representative tasks, building a BadNet attack on a Neural Network, and then running the attack on the LTN model, on at least one representative task.

## Building the LTN model

## Setup

In order to make the LTN examples run, first you need to create a virtual environment in your folder. You can do this by running the following command:

```bash
python -m venv .venv
```

Then, you need to activate the virtual environment. You can do this by running the following command:

```bash
source .venv/bin/activate
```

Then, you need to make sure you install the current version of ltn from your package with:

```bash
pip install -e .\logictensornetwork\
```

Then, you need to install the requirements. You can do this by running the following command:

```bash
pip install -r requirements.txt
```

The python version is Python 3.12, pip version is 23.2.1. By doing this setup, you can ensure that you won't encounter deprecated packages errors. You will also need to install Jupyter Notebook.



## Experiments

For the first milestone, the https://github.com/logictensornetworks/logictensornetworks repository was used as a reference. This repository contains a tutorial on how to use the LTN framework and a few examples on representative tasks for LTN https://github.com/logictensornetworks/logictensornetworks/tree/master/examples. From these examples, I chose using the **mnist** and the **smokes_friends_cancer** problems. 

Although the used repository has the latest ltn version, the examples weren't updated to run on it, so in this repository I first had to modify the ipynb files from those directories to make them runnable. The experiments that were run can be found in the [mnist directory](logictensornetwork/examples/mnist) and [smokes_friends_cancer](logictensornetwork%2Fexamples%2Fsmokes_friends_cancer). After making the code runnable, I focused more on the **mnist** problem, by running both the single_digit_addition and multi_digit_addition notebooks. Then, I  changed the operation that the model was being trained on (from addition to subtraction), experiment which can be found in [[single_digits_difference.ipynb](logictensornetwork%2Fexamples%2Fmnist%2Fsingle_digits_difference.ipynb)] .

The first experiment was a simple one, where I trained the model to learn the subtraction of two digits.

[all_experiments_mnist.ipynb](logictensornetwork%2Fexamples%2Fmnist%2Fall_experiments_mnist.ipynb) contains the more important experiments on MNIST, where I also played around with p_schedules, learning rates, number of epochs and layers.
That file also contains the observations made on the results of the experiments and a final overview of how LTN acts with more or less symbolic knowledge.
## Building the BadNet attack
The BadNet attack was done by inserting a small white square trigger on the bottom-right corner of grayscale MNIST images and changing these images target label to 1.  The implementation can be found at [badnet.py](Badnet/badnet.py).

The attack was implemented on a simple Convolutional Neural Network (CNN) model, which was trained to classify MNIST digits. The CNN used is the same one from the SingleDigit model in the ltn( which can be found at [baselines.py](logictensornetwork/examples/mnist/attacks/baselines.py)) The attack consists of the following steps: retrive the train data, insert the trigger on 10% of the images, train the model on the modified dataset, and then test the model on a fully-poisoned dataset and a clean dataset. The attack is successful if it achieves the same accuracy on the clean dataset as the original model, and a relatively high accuracy on the poisoned dataset (which is the one with the trigger inserted).
The attack was run on the MNIST dataset, and the results can be found in [badnet_resultd](Badnet/badnet_results). The results show that the attack is successful, with the model achieving a high accuracy on the poisoned dataset and a high accuracy on the clean dataset. 


## Running the BadNet on the LTN
Several configuratins of BadNets were run on the LTN model solving the Single-Digit Addition and Multi-Digit Addition tasks. 

From the basic attack presented above ( 6x6 pixels trigger
on right side of the first image), a few variations of hyperpa-
rameters were tested:

• trigger sizes of 4, 6 and 10

• the trigger position was the bottom-right corner or the center of the image

• either one image was poisoned, or both

• different number of training epochs

• different p schedules.

The results of the experiments can be found at [attacks](logictensornetwork/examples/mnist/attacks). Clean label attacks were also run, where the label of the triggered images was kept the same. However, for the purpose of the study, they were not included in the final report, but they are available in this folder. 

The results showed that:

• Larger (e.g., 6×6), centrally placed triggers are the most
effective, achieving near-100% ASR without harming
clean accuracy.

• Poisoning both images in a sample often leads to low
ASR due to symbolic ambiguity, preserving model per-
formance.

• Symbolically dominant inputs (e.g., d1 and d3 in MDA)
are more sensitive to poisoning.

• Stronger regularization hyperparameters during training
supress weaker attacks over time.

More information on the results can be found at TODO: put link to the report.

