import math
import runModel
import numpy as np
import argparse
import os
import time
import sys

parser = argparse.ArgumentParser(description='Hyperparameters for RNN training')
parser.add_argument('nPoints', type=int, help='Number of points in the trajectory (required)')
parser.add_argument('wsd', type=float, help='Standard deviation used to initialized the weight matrix')
parser.add_argument('--epochs', type=int, default=50, help='Maximum number of epochs')
parser.add_argument('--autoLR', type=bool, default=False, help='Whether or not to use pre-tuned learning rates based on model size')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate')
parser.add_argument('--nhid', type=int, default=128, help='Number of hidden features')
parser.add_argument('--isl', type=float, default=9, help='Number of points to use in training. npoints-isl is the number of points to predict')
parser.add_argument('--scaledISL', type=bool, default=False, help='If set to true, the isl variable will be interpreted as the percentage of  the total sequence to be used as input')
parser.add_argument('--save', default='savedResults/test', type=str, help='Location to store saved results')
parser.add_argument('--gpu', action='store_true', help='Use GPU?')
parser.add_argument('--cp', action='store_true', help='Enable Checkpointing. Will try to look for a checkpoint to load from.')

args = parser.parse_args()
# Create directories to save to if they don't exist:
def createDir(path):
    try:
        os.makedirs(path)
    except OSError:
            if not os.path.isdir(path):
                raise
                
createDir(args.save)
createDir('%s/TrajPlots' % args.save)
createDir('%s/LogFiles'  % args.save)
createDir('%s/TrajModel' % args.save)
createDir('%s/checkpoints' % args.save)

if args.autoLR:
    if args.nPoints <= 40:
        args.lr = 0.05
    elif args.nPoints <= 60:
        args.lr = 0.01
    elif args.nPoints < 120:
        args.lr = 0.005
    elif args.nPoints >= 120:
        args.lr = .001
# Tune the input sequence length if required
if args.scaledISL:
    if args.isl >= 1:
        raise SystemExit("Percentage of sequence as input is >= 1")
    for el in data:
        args.isl = int(nPoints*args.isl)
        print(args.isl)
# Run Model:
start = time.time()
trainLoss, testLoss, stopped_early = runModel.run("lorAtt_%d" % args.nPoints, args.wsd, args.epochs, args.lr, args.nhid, args.isl, args.save, start, args.gpu, args.cp)

print("Total runtime was: %s" % (runModel.timeSince(start)))
print(args.nPoints, args.wsd, trainLoss, testLoss)
np.save('%s/loss_%d_%0.3f.npy' % (args.save, args.nPoints, args.wsd), [args.nPoints, args.wsd, trainLoss, testLoss])
if stopped_early:
    sys.exit(123) #Indicates to Jean-Roch's code that there is no need to return to this model
try:
    os.rename('nohup.out', args.save + '/nohup.out')
except OSError:
    if os.path.isfile('nohup.out'):
        raise
        
