#!/bin/bash

for i in `seq 1`
do
    for i in `seq 1 140`
    do
        python eval_valid_feature_lgb.py $i
    done
done
