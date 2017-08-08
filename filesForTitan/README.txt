Required Packages (tried to provide the version number, but the latest version of the packages should work): 

python (2.7.12+ or 3.6+)

torch (pytorch) (0.1.12.post2)
math
numpy (1.11.0)
dill (0.2.7.1)
argparse (1.2.1)
os
time (1.7-25.1)
matplotlib (2.0.2)
random

To run experiment, open a terminal, cd into the file where the tarball was unpacked, and type:
(There will be no output, as it is all appended to the nohup.out file that is created. All data is saved into a file named by the command line argument, --save)

nohup python main.py --save 'titanRun1' --epochs 500 --autoLR True &

To run a test to make sure the environment and code is setup correctly (should take ~ 5 min):

python test.py --save 'test' --epochs 5 

To run a single model:

python single.py [Number of points in the sequence] [Standard deviation of weights] --epochs [number of epochs] --autoLR True --save [File name to save to]
The runSingle.sh file contains how to run all the models from shell.

To use checkpointing, enable the --cp flag. This will save a checkpoint for every epoch of the model to the location -> --save/checkpoints.
To resume a training, set the --resume flag to the file location of the checkpoint. Be careful to ensure the loaded checkpoint as the same
hyperparameters (mainly number of points and standard deviation of the weights).