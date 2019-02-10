#!/bin/bash -

i=10
while [ $i -lt 30 ]
do
    (( i++ ))
    ./shfl_vs_sm.out $i
done
