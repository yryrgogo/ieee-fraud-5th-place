#!/bin/bash

for i in `seq 1 10`
do
    for i in `seq 1 20`
    do
        python eval_valid_feature_lgb.py $i
    done
    mv ../feature/valid_use/* ../feature/valid/
done

