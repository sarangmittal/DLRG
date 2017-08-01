#!/bin/bash
pts=( 10 20 30 40 50 60 80 100 120 )
std=( 0.001 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10)
for i in ${pts[*]}; do
    for j in ${std[*]}; do
        echo nPoints: $i
        echo std: $j
        python single.py $i $j --epochs 500 --autoLR True --save 'titanRun1' 
    done
done
echo