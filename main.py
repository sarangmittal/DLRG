import math
import runModel
import numpy as np
import pickle
import argparse
import os


parser = argparse.ArgumentParser(description='Hyperparameters for RNN training')
parser.add_argument('--epochs', type=int, default=50, help='Maximum number of epochs')
parser.add_argument('--lr', type=float, default=0.005, help='Learning Rate')
parser.add_argument('--nhid', type=int, default=128, help='Number of hidden features')
parser.add_argument('--isl', type=int, default=9, help='Number of points to use in training. npoints-isl is the number of points to predict')
parser.add_argument('--save', type=str, help='Location to store saved results')

args = parser.parse_args()
nPoints = [10,20,30,40,50,60]
wsd2 = [0.001, 0.005, 0.01, 0.015, 0.02]

# Create directories to save to if they don't exist:
def createDir(path):
    try:
        os.makedirs(path)
    except OSError:
            if not os.path.isdir(path):
                raise

createDir(args.save)
createDir('%s/TrajPlots' % args.save)
createDir('%s/TrajModel' % args.save)

# Write an info file
with open('%s/info.txt' % args.save, 'w') as f:
    f.write('This is the information file containing the hyperparameters\n')
    f.write('Number of Epochs: %d \n' % args.epochs)
    f.write('Learning Rate: %f \n' % args.lr)
    f.write('Number of features in the hidden state: %d \n' % args.nhid)
    f.write('Number of points that go into encoder: %d \n' % args.isl)
    f.write('The model is a sequence 2 sequence RNN, trained using the MSE Loss function, \n SGD Optimizer.\n')
    f.write('The range of sequence lengths is:\n')
    for n in nPoints:
        f.write('%d\n' % n)
    f.write('The range of standard deviations squared of the weight init is:\n')
    for w in wsd2:
        f.write('%f\n' % w)
    f.write('Code written by Sarang Mittal (Caltech), using the Pytorch API')
    f.close()


allError = np.zeros((len(nPoints) + 1, len(wsd2) + 1))
# The matrix will have labels of the number of points and sigma squared

for w in range(len(wsd2)):
    allError[0][w+1] = wsd2[w]
for n in range(len(nPoints)):
    allError[n + 1][0] = nPoints[n]
    for w in range(len(wsd2)):
        print('*****************************************************************************')
        print('Starting Model with nPoints = %d and weight standard deviation squared = %.3f' % (nPoints[n], wsd2[w]))
        print('*****************************************************************************')
        trainError = runModel.run('lorAtt_%d' % nPoints[n], math.sqrt(wsd2[w]), args.epochs, args.lr, args.nhid, args.isl, args.save)
        print('************  Train Error is %.4g   ******************************************' % trainError)
        allError[n + 1][w + 1] = trainError
        print('*****************************************************************************')
        print('Finished Model with nPoints = %d and weight standard deviation squared = %.3f' % (nPoints[n], wsd2[w]))
        print('*****************************************************************************')
with open('%s/allTrainLosses.pickle' % args.save, 'wb') as f:
    pickle.dump(allError, f)
    