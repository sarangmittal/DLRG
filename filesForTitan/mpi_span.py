import os
import itertools
import random
import glob
import time
import sys

ptss = map(int,sys.argv[1].split(','))
vars = map(float, sys.argv[2].split(','))
epoch = int(sys.argv[3])


dir='titanMpiRun_Aug21'
if len(sys.argv)>4:
    dir = sys.argv[4]

use_gpu = False

combinations = list(itertools.product(ptss, vars))
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
                print ("No more task left, idling worker %d at %s"%( source, time.asctime(time.localtime())))

            else:
                a_task = random.choice( tasks )
                executing.append( a_task )
                tasks.remove( a_task )
                comm.send(a_task, dest=source, tag=START)
                print("Sending task %d to worker %d at %s"% (a_task, source, time.asctime(time.localtime())))
        elif tag == DONE:
            result = int(data)
            print("Got data from worker %d at %s" %( source, time.asctime(time.localtime())))
            tasks.append( result )
            executing.remove( result )
        elif tag == NOTRAIN:
            result = data
            print ("Worker %d considers %d as done at %s"%(source , result,time.asctime(time.localtime())))
        elif tag == EXIT:
            print("Worker %d exited at %s" % (source,time.asctime(time.localtime())))
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
            pts, var = combinations[int(task)]
            com = "python single.py %s %s --epochs %d --autoLR True --save '%s' %s --cp --nhid 256 --isl 9 "%( 
                pts, 
                var, 
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



comm.Barrier() # wait for everybody to synchronize _here_
