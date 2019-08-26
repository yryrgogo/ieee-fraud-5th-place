import os
from pathlib import Path
import shutil


HOME = Path().home()
DIR_PJ = HOME / 'github/ieee-fraud'


def move_feature(list_feature, from_dir, to_dir):
    
    DIR_FROM_FEATURE = Path(f'feature/{from_dir}') 
    DIR_TO_FEATURE   = Path(f'feature/{to_dir}') 
    
    for feature in list_feature:
        from_train_feature_path = str(DIR_PJ / DIR_FROM_FEATURE / f'{feature}_train.gz')
        from_test_feature_path  = str(DIR_PJ / DIR_FROM_FEATURE / f'{feature}_test.gz')
        to_path = str(DIR_PJ / DIR_TO_FEATURE)
        
        shutil.move(from_train_feature_path, to_path)
        shutil.move(from_test_feature_path, to_path)
