import os
import sys
import time

n_workers = 50
ptss = [ 10, 20, 30, 40, 50, 60, 80, 100, 120]
vars = [ 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6 ]
epoch=500
dir= 'titanMpiRun_Aug21'

n_tasks = len( ptss) * len(vars)
print (n_tasks,"tasks to run, with",n_workers,"workers")
if n_workers > n_tasks:
    sys.exit( 10 )

walltime = 2*60
if n_workers > 125:
    walltime = 6*60

mpi_script = 'mpi_{0:d}.pbs'.format( n_workers )
mpi_file = open(mpi_script,'w')
mpi_file.write("""#!/bin/bash
#    Begin PBS directives
#PBS -A hep107
#PBS -N mpi_run_{1:d}
#PBS -j oe
####PBS -q debug
#PBS -l walltime={0:d}:00,nodes={1:d}
#    End PBS directives and begin shell commands

cd $MEMBERWORK/hep107

export HOME=/lustre/atlas/scratch/vlimant/hep107/
export PYTHONPATH=/ccs/proj/hep107/sft/lib/python3.5/site-packages/

module load python/3.5.1
module load python_mpi4py

cd /lustre/atlas/proj-shared/hep107/DLRG/filesForTitan/
date
aprun -n {1:d} -N 1 python mpi_span.py {2} {3} {4:d} {5}
""".format(
        walltime, 
        n_workers,
        ','.join( map(str,ptss) ),
        ','.join( map(str,vars) ),
        epoch,
        dir
        ))

mpi_file.close()
current_mpi_job_id = sys.argv[1] if len(sys.argv)>1 else None
n_submissions = 10
while n_submissions>0:
    if current_mpi_job_id != None:
        while os.popen('showq -u vlimant | grep %s '% current_mpi_job_id).read():
            print ("Job %s is still active, keep waiting"% current_mpi_job_id)
            time.sleep( 60 )

    n_submissions -= 1
    print (time.asctime( time.localtime()))
    print ("Submit the mpi job")
    sub = os.popen('qsub %s'%mpi_script)
    current_mpi_job_id = int(sub.read())
    time.sleep(60) ##it takes some time to get the job id in showq
