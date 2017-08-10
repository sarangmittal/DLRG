import os
import itertools
import random
import glob
import time


ptss = [ 10, 20, 30, 40, 50, 60, 80, 100, 120]
#stds = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
stds = [0.1, 0.5, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6]

epoch=100
dir='titanRun'

>>>>>>> mpi drivers
use_gpu = False

combinations = list(itertools.product(ptss, stds))
print (len(combinations))
print (combinations[0])

## the mpi part
from mpi4py import MPI
comm = MPI.COMM_WORLD

size = comm.size        # total number of processes
rank = comm.rank        # rank of this process
status = MPI.Status()   # get MPI status object

## tags
READY=0
DONE=1
EXIT=2
START=3
NOTRAIN=4


if rank == 0:
    tasks = list(range( len(combinations)))
    executing = list()

    task_index = 0
    num_workers = size - 1 # remove the master
    closed_workers = 0
    tasks_done = set()
    print("Master starting with %d workers" % num_workers)
    while closed_workers < num_workers:
        data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        source = status.Get_source()
        tag = status.Get_tag()
        if tag == READY:
            # Worker is ready, so send it a task
            ### keep on running until running out of time
            if not tasks:
                ## we have to assume that there is more overall tasks than workers
                comm.send(None, dest=source, tag=EXIT)
                print ("No more task left, idling worker %d"% source)
            else:
                a_task = random.choice( tasks )
                executing.append( a_task )
                tasks.remove( a_task )
                comm.send(a_task, dest=source, tag=START)
                print("Sending task %d to worker %d"% (a_task, source))
                time.sleep(10)
        elif tag == DONE:
            result = data
            print("Got data from worker %d" % source)
            tasks.append( result )
            executing.remove( result )
        elif tag == NOTRAIN:
            result = data
            print ("Worker %d considers %d as done"%(source , result))
        elif tag == EXIT:
            print("Worker %d exited." % source)
            closed_workers += 1

    print("Master finishing")
else:
    # Worker processes execute code below
    name = MPI.Get_processor_name()
    print("I am a worker with rank %d on %s." % (rank, name))
    while True:
        comm.send(None, dest=0, tag=READY)
        task = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()
        
        if tag == START:
            print ("Received index",task,"to operate on")
            # Do the work here
            pts, std = combinations[int(task)]
            com = "python single.py %s %s --epochs %d --autoLR True --save '%s' %s --cp --nhid 128 --isl 9 "%( 
                pts, 
                std, 
                epoch, 
                dir ,
                '--gpu' if use_gpu else '')
            
            print ("Will execute the command",com)
            #code = random.choice( [0]*5 + [123]*1 ) ## fake a 1/6 finishing rate
            code = os.system(com)
            ## is there a way to catch that single.py exited without running a single epoch ? yes exit code 123
            result = int(task)
            comm.send(result, dest=0, tag=DONE if code==0 else NOTRAIN)
        elif tag == EXIT:
            break

    comm.send(None, dest=0, tag=EXIT)



"""
if comm.rank< len(combinations):
    pts, std = combinations[comm.rank]
    com = "python single.py %s %s --epochs %d --autoLR True --save '%s' %s"%( pts, std, epoch, dir ,'--gpu' if use_gpu else '')
    print ("Will execute the command",com)
    ##os.system(com)
else:
    print ('nothing for me to do')
"""

comm.Barrier() # wait for everybody to synchronize _here_
