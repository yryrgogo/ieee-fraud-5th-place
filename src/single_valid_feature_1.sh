#!/bin/bash

for i in `seq 1 50`
do
    # python eval_lgb_downsampling_single_valid_feature.py 1
    python eval_lgb_bear_single_valid_feature.py 1
done

wait
sudo shutdown -P