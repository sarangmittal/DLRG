import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from torch.nn import init
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import numpy as np
import random
import math
import time
import copy

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, weight_std):
        super(RNN, self).__init__()
        # Initialization constants
        self.weight_std = weight_std
        self.weight_mean = 0.0
        self.bias_mean = 0.0
        self.bias_std = math.sqrt(0.05)
        self.hidden_size = hidden_size
        # Modules
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.activation = nn.LeakyReLU(0.01)
        # Weight Initialization
        for m in self.modules():
                if isinstance(m, torch.nn.Linear):
                    m.weight.data.normal_(self.weight_mean, self.weight_std)
                    m.bias.data.normal_(self.bias_mean, self.bias_std)
        
    def forward(self, input, hidden):
        combined = torch.cat([input, hidden], 1)
        hidden = self.i2h(combined)
        hidden = self.activation(hidden)
        output = self.i2o(combined)
        return output, hidden
    
    def initHidden(self):
        return Variable(torch.zeros(1,self.hidden_size)) #.cuda() # if use_cuda else Variable(torch.zeros(1,self.hidden_size))

# Function to train model
def train(input_batch, encoderRNN, decoderRNN, encoder_optimizer, decoder_optimizer, input_sequence_length, criterion, n_dim, batch_size):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0
    for c in range(batch_size):
        input_traj = input_batch[c]
        decoderOutput, singleLoss = evaluate(input_traj, encoderRNN, decoderRNN, input_sequence_length, criterion, n_dim, True)
        loss += singleLoss

    loss /= batch_size
    loss.backward()
    # Update parameters
    encoder_optimizer.step()
    decoder_optimizer.step()
    
    return loss.data[0]

# Function for evaluating on validation and test sets
def evaluate(input_traj, encoderRNN, decoderRNN, input_sequence_length, criterion, n_dim, full_loss):
    hidden = encoderRNN.initHidden()
    input_traj = Variable(input_traj) # if use_cuda else Variable(circle)    

    output_sequence_length = input_traj.size()[0] - input_sequence_length
    
    encoderOutput = Variable(torch.zeros(input_sequence_length, 1, n_dim)) #if use_cuda else encoderOutput
    decoderOutput = Variable(torch.zeros(output_sequence_length,1, n_dim)) #if use_cuda else decoderOutput
    
    # Run the point sequence into the encoder
    for i in range(input_sequence_length):
        encoderOutput[i], hidden = encoderRNN(input_traj[i], hidden)
    
    # Now the last hidden state of the encoder is the first hidden state of the decoder. (whats the input to the first)
    # For now, let's have the first input be the origin
    for i in range(output_sequence_length):
        if (i == 0):
            dummyState = Variable(torch.zeros(1,n_dim)) # if use_cuda else Variable(torch.zeros(1,n_dim))
            decoderOutput[i], hidden = decoderRNN(dummyState, hidden)
        else:
            decoderOutput[i], hidden = decoderRNN(decoderOutput[i-1], hidden)
    
    loss = criterion(decoderOutput, input_traj[-(output_sequence_length):]) #.cuda())
    if full_loss:
        return decoderOutput, loss 
    else:
        return decoderOutput, loss.data[0]


# Keep track on time
def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s/60)
    s -= m * 60
    return '%dm %ds' % (m,s)


def run(data_string, weigh_var, n_epochs, learning_rate, hidden_features, input_sequence_length, save_location, start):
    # Constants
    use_cuda = torch.cuda.is_available()
    #earlyStoppingCriteria = 10
    batch_size = 4
    wsd = np.sqrt(weigh_var/hidden_features)
    # Load in data
    with open('lorAttData/%s.pickle' % (data_string), 'rb') as f:
        data = list(pickle.load(f))
    # data = list(np.load('lorAttData/npy/%s.npy' % (data_string)))
    # Open a file to write output to:
    logFile = open('%s/LogFiles/log_%s_%.3g.txt' % (save_location, data_string, weigh_var), 'w')

    # Partition Data into Training, Validation, and Test sets (80/10/10)
    random.seed(12345) # Comment out to get a different split every time
    random.shuffle(data)
    training = data[:(len(data)/10 * 8)]
    val = data[(len(data)/10 * 8):(len(data)/10 * 9)]
    test = data[(len(data)/10 * 9):]
    
    # Convert data to torch tensors
    for i in range(len(training)):
        training[i] = torch.FloatTensor(training[i])
    for i in range(len(val)):
        val[i] = torch.FloatTensor(val[i])
    for i in range(len(test)):
        test[i] = torch.FloatTensor(test[i])
        
    # Create the model
    n_dim = training[0].size()[2]

    encoderRNN = RNN(n_dim, hidden_features, n_dim, weigh_var)
    decoderRNN = RNN(n_dim, hidden_features, n_dim, weigh_var)
    
    criterion = nn.MSELoss(size_average = True)

    encoder_optimizer = optim.SGD(encoderRNN.parameters(), lr = learning_rate)
    decoder_optimizer = optim.SGD(decoderRNN.parameters(), lr = learning_rate)
    
    # Storage variables for training
    current_loss_train = 0
    current_loss_val = 0
    all_losses_train = []
    all_losses_val = []
    trainLen = len(training)
    valLen = len(val)
    
    # Start Training
    logFile.write('*****************************************************************************\n')
    logFile.write('Starting Model with nPoints = %d and weight variance = %.3g\n' % (training[0].size()[0], weigh_var))
    logFile.write('*****************************************************************************\n')
    
    for ep in range(n_epochs):
        n_batches = len(training)/batch_size
        for i in range(n_batches):
            batch = training[i*batch_size:((i+1)*batch_size)]
            loss = train(batch, encoderRNN, decoderRNN, encoder_optimizer, decoder_optimizer, 
                         input_sequence_length, criterion, n_dim, batch_size)
            current_loss_train += loss/n_batches

        for c in range(len(val)):
            pts, loss = evaluate(val[c], encoderRNN, decoderRNN, input_sequence_length, criterion, n_dim, False)
            current_loss_val += loss/valLen
        if (ep % 100 == 0):
            print("Finished epoch [%d / %d] on model with %d points and variance %.3f" % (ep + 1, n_epochs,training[0].size()[0], weigh_var))
            
        #Early Stopping
        # Don't need Early Stopping, as we are trying to show trainability
        # # Method: Stop if Generalization Loss > 10% (see Automatic Early Stopping Using Cross Validation: Quantifying the Criteria by
        # # Lutz Prechelt)
        # if(ep == 0):
        #     bestEncoder = copy.deepcopy(encoderRNN)
        #     bestDecoder = copy.deepcopy(decoderRNN)
        # if ((ep != 0)):
        #     if (current_loss_val <= min(all_losses_val)):
        #         #print("New min at epoch %d" %(ep+1))
        #         bestEncoder = copy.deepcopy(encoderRNN)
        #         bestDecoder = copy.deepcopy(decoderRNN)
        #     gLoss = current_loss_val * (1/min(all_losses_val)) * 100 - 100
        #     #print("Generalization Loss at Epoch %d is %.3g" % (ep + 1, gLoss))
        #     if (gLoss > earlyStoppingCriteria):
        #         print("Stopping Early on epoch %g" % (ep+1))
        #         print("Generalization Loss at Epoch %d is %.3g > %g" % (ep + 1, gLoss, earlyStoppingCriteria))
        #         #print("Validation loss was %.4g" % (current_loss_val/len(val)))
        #         encoderRNN = bestEncoder
        #         decoderRNN = bestDecoder
        #         break

        all_losses_train.append(current_loss_train)
        current_loss_train = 0
        all_losses_val.append(current_loss_val)
        current_loss_val = 0
        logFile.write("Finished epoch [%d / %d]  with training loss %.4g and validation loss %.4g\n" 
              % (ep + 1, n_epochs, all_losses_train[ep], all_losses_val[ep]))
        logFile.write(timeSince(start) + "\n")
        if math.isnan(all_losses_train[-1]):
            logFile.write("Stopped because loss exploded to NaN and will not converge\n")
            break
        # Method 3: Stop if Progress < 1. Essentially, stop if the model has converged.
        strip_length = 5
        if ((ep >= (strip_length - 1))):
            last_strip = all_losses_train[-strip_length:]
            progress = (np.mean(last_strip)/min(last_strip) - 1) * 1000
            logFile.write("Progress was %.3f\n" % progress)
            if progress < 1:
                logFile.write("Stopping early on epoch %d\n" % (ep + 1))
                break
    
    logFile.write('************  Train Error is %.4g   ******************************************\n' % all_losses_train[-1])


    
    # Plot and save training and validation loss
    #plt.figure()
    plt.plot(all_losses_train, 'r', label="Training Loss")
    plt.plot(all_losses_val, 'b', label="Validation Loss")
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss (log scale)')
    plt.yscale('log')
    plt.title('Loss over Epochs on %s set with Weight sigma = %.3g ' % (data_string, weigh_var))
    plt.legend()
    plt.savefig('%s/TrajPlots/loss_data_%s_wsd_%.3g.png' % (save_location, data_string, weigh_var))
    plt.close()
    
    # Plot model output on an example circle and save
    from mpl_toolkits.mplot3d import Axes3D
    testTraj = test[random.randint(0,len(test))]
    start_traj = testTraj.cpu().numpy()
    points, loss = evaluate(testTraj, encoderRNN, decoderRNN, input_sequence_length, criterion, n_dim, False)
    end_traj = points.data.numpy()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(start_traj[0:1,0,0], start_traj[0:1,0,1], start_traj[0:1,0,2], 'gs', label='Start')
    ax.plot(start_traj[:,0,0], start_traj[:,0,1], start_traj[:,0,2], 'r', label='Ground Truth')
    ax.plot(end_traj[:,0,0], end_traj[:,0,1], end_traj[:,0,2], 'b.', label='Predictions')
    plt.legend()
    plt.title("Model Prediction on Example from %s and Weight sigma = %.3g" %(data_string, weigh_var))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('%s/TrajPlots/vis_data_%s_wsd_%.3g.png' % (save_location, data_string, weigh_var))
    plt.close()
    # Save the model
    torch.save(encoderRNN.state_dict(), '%s/TrajModel/%s_encoder_wsd_%.3g' % (save_location, data_string, weigh_var))
    torch.save(decoderRNN.state_dict(), '%s/TrajModel/%s_decoder_wsd_%.3g' % (save_location, data_string, weigh_var))
    
    # Calculate the test loss:
    test_loss = 0
    testLength = len(test)
    for t in range(len(test)):
        pts, loss = evaluate(test[t], encoderRNN, decoderRNN, input_sequence_length, criterion, n_dim, False)
        test_loss += loss/testLength
    logFile.write('************  Test Error is %.4g *********************************************\n' % test_loss)
    logFile.write('*****************************************************************************\n')
    logFile.write('Finished Model with nPoints = %d and weight variance = %.3g\n' % (training[0].size()[0], weigh_var))
    logFile.write('*****************************************************************************\n')
    logFile.close()
    return all_losses_train[-1], test_loss # Return the last error of the training and test error
