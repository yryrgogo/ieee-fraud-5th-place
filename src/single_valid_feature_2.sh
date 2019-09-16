#!/bin/bash

for i in `seq 1 50`
do
    python eval_lgb_downsampling_single_valid_feature.py 2
done
