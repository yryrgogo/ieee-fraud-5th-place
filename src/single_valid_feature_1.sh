#!/bin/bash

for i in `seq 1 50`
do
    python eval_single_valid_feature_lgb.py 4 1
done
