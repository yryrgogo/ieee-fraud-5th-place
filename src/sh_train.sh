#!/bin/bash

for i in `seq 1 5`
do
    for i in `seq 1 3`
    do
        python eval_valid_feature_lgb.py $i
    done
    mv ../feature/valid_use/* ../feature/valid/
done

