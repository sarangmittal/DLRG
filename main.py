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
parser.add_argument('--customLR', type=bool, default=False, help='Whether or not to use pre-tuned learning rates based on model size')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate')
parser.add_argument('--nhid', type=int, default=128, help='Number of hidden features')
parser.add_argument('--isl', type=int, default=9, help='Number of points to use in training. npoints-isl is the number of points to predict')
parser.add_argument('--save', type=str, help='Location to store saved results')

args = parser.parse_args()

# nPoints = [10]
# wsd = [0.01,0.02]
nPoints = [60]
wsd = [0.01]

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

# Write an info file
with open('%s/info.txt' % args.save, 'w') as f:
    f.write('This is the information file containing the hyperparameters\n')
    f.write('Number of Epochs: %d \n' % args.epochs)
    # f.write('Learning Rate: %f \n' % args.lr)
    if args.customLR:
        f.write('Learning rate is 0.05 for sequences <= 40, and 0.01 for sequences > 40 and 0.005 for sequences > 60 \n')
    else:
        f.write('Learning Rate: %f \n' % args.lr)

    f.write('Number of features in the hidden state: %d \n' % args.nhid)
    f.write('Number of points that go into encoder: %d \n' % args.isl)
    f.write('The model is a sequence 2 sequence RNN, trained using the MSE Loss function, \n SGD Optimizer.\n')
    f.write('The LeakyReLU non-linearity is applied (negative slope of 0.01) \n')
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
    print("Starting models with %d points" % nPoints[n])
    allTrainError[n + 1][0] = nPoints[n]
    allTestError[n + 1][0] = nPoints[n]
    if args.customLR:
        if nPoints[n] <= 40:
            args.lr = 0.05
        elif nPoints[n] <= 60:
            args.lr = 0.01
        elif nPoints[n] > 60:
            args.lr = 0.005
    print(args.lr)
    nWorkers = min(5, len(wsd))
    pool = mp.Pool(nWorkers) 
    temp = pool.map(runModel.run, zip(['lorAtt_%d' % nPoints[n]]*len(wsd), wsd, [args.epochs]*len(wsd), [args.lr]*len(wsd),
                                      [args.nhid]*len(wsd), [args.isl]*len(wsd), [args.save]*len(wsd), [start]*len(wsd)))
    pool.close()
    pool.join()
    for i in range(len(wsd)):
        allTrainError[n+1][i+1] = temp[i][0]
        allTestError[n+1][i+1] = temp[i][1]
    print("Finished models with %d points" % nPoints[n])
    
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