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

To run a single model:

python single.py [Number of points in the sequence] [Standard deviation of weights] --epochs [number of epochs] --autoLR True --save [File name to save to]
The runSingle.sh file contains how to run all the models from shell.

To use checkpointing, enable the --cp flag. This will save a checkpoint for last epoch of the model to the location -> --save/checkpoints.
To resume a training, set the --cp flag and the code looks for checkpoints created by this code.

The optional arguments desired are:
--epochs 500 --autoLR True --nhid 256 --isl 9 --save 'titanRun1' --cp
(you might want to try running solely on the CPU, as I saw better performance on the CPU rather than the GPU)
# Points in Sequence:
[10 20 30 40 50 60 80 100 120]
Weight var:
[0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0 2.2 2.4 2.6]
