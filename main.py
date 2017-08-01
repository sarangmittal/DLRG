import math
import runModel
import numpy as np
import pickle
import argparse
import os
import torch.multiprocessing as mp
import time

parser = argparse.ArgumentParser(description='Hyperparameters for RNN training')
parser.add_argument('--epochs', type=int, default=50, help='Maximum number of epochs')
parser.add_argument('--autoLR', type=bool, default=False, help='Whether or not to use pre-tuned learning rates based on model size')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate')
parser.add_argument('--nhid', type=int, default=128, help='Number of hidden features')
parser.add_argument('--isl', type=float, default=9, help='Number of points to use in training. npoints-isl is the number of points to predict')
parser.add_argument('--scaledISL', type=bool, default=False, help='If set to true, the isl variable will be interpreted as the percentage of  the total sequence to be used as input')
parser.add_argument('--save', type=str, help='Location to store saved results')

args = parser.parse_args()

nPoints = [20, 40, 80]
wsd = [0.01, 0.04, 0.07]
# nPoints = [10,20,30,40,50,60,80,100]
# wsd = [0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
nPoints_i = [(nPoints[i],i) for i in range(len(nPoints))]
wsd_i = [(wsd[i],i) for i in range(len(wsd))]

# Create directories to save to if they don't exist:
def createDir(path):
    try:
        os.makedirs(path)
    except OSError:
            if not os.path.isdir(path):
                raise

# Proxy function to unpack tuple and runmodel
def proxy(param):
    param = tuple(param)
    x, epochs, lr, nhid, isl, save, startTime = param
    pts_info, std_info = x
    pts_index, pts = pts_info
    std_index, std = std_info
    trainLoss, testLoss = runModel.run("lorAtt_%d" % pts, std, epochs, lr, nhid, isl, save, startTime)
    return pts_index, std_index, trainLoss, testLoss

createDir(args.save)
createDir('%s/TrajPlots' % args.save)
createDir('%s/LogFiles'  % args.save)
createDir('%s/TrajModel' % args.save)

# Write an info file
with open('%s/info.txt' % args.save, 'w') as f:
    f.write('This is the information file containing the hyperparameters\n')
    f.write('Number of Epochs: %d \n' % args.epochs)
    # f.write('Learning Rate: %f \n' % args.lr)
    if args.autoLR:
        f.write('Learning rate is 0.05 for sequences <= 40, and 0.01 for sequences > 40 and 0.005 for sequences < 120, 0.001 for >= 120 \n')
    else:
        f.write('Learning Rate: %f \n' % args.lr)

    f.write('Number of features in the hidden state: %d \n' % args.nhid)
    f.write('The model is a sequence 2 sequence RNN, trained using the MSE Loss function, \n SGD Optimizer.\n')
    f.write('The LeakyReLU non-linearity is applied (negative slope of 0.01) \n')
    if args.scaledISL:
        f.write('The encoder takes %f percent of the sequence as input\n' % (args.isl * 100))
    else:
        f.write('The encoder takes %d points as input' % args.isl)
    f.write('The range of sequence lengths is:\n')
    for n in nPoints:
        f.write('%d\n' % n)
    f.write('The range of standard deviations of the weight init is:\n')
    for w in wsd:
        f.write('%f\n' % w)
    f.write('Code written by Sarang Mittal (Caltech), using the Pytorch API')
    f.close()

allTrainError = np.zeros((len(nPoints) + 1, len(wsd) + 1))
allTestError = np.zeros((len(nPoints) + 1, len(wsd) + 1))
# The matrix will have labels of the number of points and sigma squared
for w in range(len(wsd)):
    allTrainError[0][w+1] = wsd[w]
    allTestError[0][w+1] = wsd[w]
start = time.time()

for n in range(len(nPoints)):
    allTrainError[n + 1][0] = nPoints[n]
    allTestError[n + 1][0] = nPoints[n]

# Prepare collection of model parameters
nPoints_i = [(i, nPoints[i]) for i in range(len(nPoints))]
wsd_i = [(i, wsd[i]) for i in range(len(wsd))]
nModels = len(nPoints)*len(wsd)
data = [list(a) for a in zip([(i,j) for j in wsd_i for i in nPoints_i], [args.epochs]*nModels, [args.lr]*nModels, [args.nhid]*nModels, 
           [args.isl]*nModels, [args.save]*nModels, [start]*nModels)]
# Tune the learning rate if told to do so by command line
if args.autoLR:
    for el in data:
        pts = el[0][0][1]
        if pts <= 40:
            el[2] = 0.05
        elif pts <= 60:
            el[2] = 0.01
        elif pts < 120:
            el[2] = 0.005
        elif pts >= 120:
            el[2] = 0.001
# Tune the input sequence length if required
if args.scaledISL:
    if args.isl >= 1:
        raise SystemExit("Percentage of sequence as input is >= 1")
    for el in data:
        pts = el[0][0][1]
        el[4] = int(pts*args.isl)
        print(el[4])

#Prepare pool of workers
nWorkers = min(nModels,6) # Need to limit number of threads when running on passed-pwn
# nWorkers = nModels # Give each model its own worker on the Titan
pool = mp.Pool(nWorkers) 
# Start pool of workers on jobs
output = pool.map_async(proxy, data).get()
pool.close()
pool.join()

#Write output to correct places
for el in output:
    x, y, trainLoss, testLoss = el
    allTrainError[x+1][y+1] = trainLoss
    allTestError[x+1][y+1] = testLoss
    
print(allTrainError)
print(allTestError)
    
with open('%s/allTrainLosses.pickle' % args.save, 'wb') as f:
    pickle.dump(allTrainError, f)
with open('%s/allTestLosses.pickle' % args.save, 'wb') as f:
    pickle.dump(allTestError, f)
print("Total runtime was: %s" % (runModel.timeSince(start)))
try:
    os.rename('nohup.out', args.save + '/nohup.out')
except OSError:
    if os.path.isfile('nohup.out'):
        raise