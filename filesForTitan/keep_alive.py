import os
import sys
import time

current_mpi_job_id = sys.argv[1] if len(sys.argv)>1 else None
n_submissions = 10
while n_submissions>0:
    if current_mpi_job_id != None:
        while os.popen('showq -u vlimant | grep %s '% current_mpi_job_id).read():
            print ("Job %s is still active, keep waiting"% current_mpi_job_id)
            time.sleep( 60 )

    n_submissions -= 1
    print ("Submit the mpi job")
    sub = os.popen('qsub mpi.pbs')
    current_mpi_job_id = int(sub.read())
