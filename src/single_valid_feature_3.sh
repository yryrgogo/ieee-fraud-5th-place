#!/bin/bash

for i in `seq 1 600`
do
#     python eval_lgb_downsampling_single_valid_feature.py 3
    python eval_lgb_bear_single_valid_feature.py 3
done
