# DLRG
Code/Data/Plots for Sarang Mittal (Caltech) DL > RG research project

Dependencies:

torch (Pytorch) (http://pytorch.org/)
matplotlib
dill/pickle
numpy
random
math
time
copy
argparse
os

File required to run an experiment:

Data Files: lorAttData folder
main.py
runModel.py

How to run experiment:
Go into main.py, and set the nPoints and wsd vectors to the desired values.
Change the arguments of the runModel.run call to select the correct data base (either lorAtt or lorAttv2)

In command line:
nohup python main.py --save 'Location to save data' --epochs (Number of epochs) --autoLR (True or False) --lr (Learning Rate) --nhid (Number of hidden features) --isl (Number of inputs to encoder. Should be shorter than shortest trajectory in dataset) &

This will run the experiment in the background and append any output to nohup.out, which can be found in the save location at the end of the experiment.