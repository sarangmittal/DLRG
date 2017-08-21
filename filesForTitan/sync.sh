#!/bin/bash

src=$1
while [ 1 ] ; do
    rsync -rv --chmod=g+rwx $src culture-plate-sm.hep.caltech.edu:/bigdata/shared/DLRG/
    sleep 5m
done