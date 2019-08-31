import numpy as np
import pandas as pd
import multiprocessing
import gc
from scipy.stats.mstats import mquantiles
from tqdm import tqdm
import sys
import os
HOME = os.path.expanduser('~')
from func.utils import parallel_process, round_size
import concurrent.futures

class Xray_Cal:

    def __init__(self, ignore_list=[], is_viz=False):
        self.ignore_list = ignore_list
        self.df_xray=[]
        self.xray_list=[]
        self.point_dict = {}
        self.N_dict = {}
        self.fold_num = None
        self.is_viz = is_viz

    def parallel_xray_calculation(self, args):
        col = args[0]
        val = args[1]
        N = args[2]
        dataset = self.df_xray.copy()
        dataset[col] = val
        pred = self.model.predict(dataset)
        p_avg = np.mean(pred)

        if self.is_viz:
            print(f'''
#========================================================================
# FOLD{self.fold_num} CALCULATION... COLUMN: {col} | VALUE: {val} | X-RAY: {p_avg}
#========================================================================''')
        del dataset
        gc.collect()
        return {'feature':col,
                'value'  :val,
                'xray'   :p_avg,
                'N'      :N
                }



    def single_xray_calculation(self, col, val, N):

        dataset = self.df_xray.copy()
        dataset[col] = val
        pred = self.model.predict(dataset)
        gc.collect()
        p_avg = np.mean(pred)

        if self.is_viz:
            print(f'''
#========================================================================
# CALCULATION PROGRESS... COLUMN: {col} | VALUE: {val} | X-RAY: {p_avg}
#========================================================================''')
        return {
            'feature':col,
            'value'  :val,
            'xray'   :p_avg,
            'N'      :N
        }


    def get_xray(self, base_xray, fold_num, col_list=[], max_point=20, N_sample=400000, ex_feature_list=[], parallel=False, cpu_cnt=multiprocessing.cpu_count()):
        '''
        Explain:
        Args:
            fold_num  : CVの場合、外部からfold番号をこのメソッドに渡す必要がある
            col_list  : x-rayを出力したいカラムリスト.引数なしの場合はデータセットの全カラム
            max_point : x-rayを可視化するデータポイント数
            ex_feature: データポイントの取得方法が特殊なfeature_list
        Return:
        '''
        result = pd.DataFrame([])
        self.fold_num = fold_num

        if len(col_list)==0:
            col_list = [col for col in base_xray.columns if col not in self.ignore_list]
        for i, col in enumerate(col_list):

            # CVなどで複数モデルの結果を平均する場合は、最初のみ行う処理
            if fold_num==0:
                null_values = base_xray[col][base_xray[col].isnull()].values
                if len(null_values)>0:
                    null_value = null_values[0]
                    null_cnt = len(null_values)

                df_not_null = base_xray[~base_xray[col].isnull()]

                #========================================================================
                # Get X-RAY Data Point
                # 1. 対象カラムの各値のサンプル数をカウントし、割合を算出。
                # 2. 全体においてサンプル数の少ない値は閾値で切ってX-RAYを算出しない
                #========================================================================
                val_cnt = df_not_null[col].value_counts().reset_index().rename(columns={'index':col, col:'cnt'})

                # 初回ループで全量データを使いデータポイント(bin)と各binのサンプル数を取得する
                # max_pointよりnuniqueが大きい場合、データポイントはmax_pointにする
                # binによる中央値と10パーセンタイルをとり, 分布全体のポイントを取得できるようにする
                if len(val_cnt)>max_point:
                    # 1. binにして中央値をとりデータポイントとする
                    bins = max_point-10
                    tmp_points = pd.qcut(x=df_not_null[col], q=bins, duplicates='drop')
                    tmp_points.name = f'bin_{col}'
                    tmp_points = pd.concat([tmp_points, df_not_null[col]], axis=1)
                    # 各binの中央値をデータポイントとする
                    mode_points = tmp_points[[f'bin_{col}', col]].groupby(f'bin_{col}')[col].median().to_frame()
                    # 各binのサンプル数を計算
                    data_N = tmp_points[[f'bin_{col}', col]].groupby(f'bin_{col}')[col].size().rename(columns={col:'N'})
                    mode_points['N'] = data_N

                    # 2. binの中央値と合わせ、percentileで10データポイントとる
                    percentiles = np.linspace(0.05, 0.95, num=10)
                    percentiles_points = mquantiles(val_cnt[col].values, prob=percentiles, axis=0)
                    max_val = df_not_null[col].max()
                    min_val = df_not_null[col].min()
                    # 小数点以下が大きい場合、第何位までを計算するか取得して丸める
                    r = round_size(max_val, max_val, min_val)
                    percentiles_points = np.round(percentiles_points, r)
                    # data point
                    data_points = list(np.hstack((mode_points[col], percentiles_points)))
                    # data N
                    data_N = list(np.hstack((mode_points['N'], np.zeros(len(percentiles_points))+np.nan )))
                else:
                    length = len(val_cnt)
                    data_points = list(val_cnt.head(length)[col].values) # indexにデータポイント, cntにサンプル数が入ってる
                    data_N = list(val_cnt['cnt'].head(length).values) # indexにデータポイント, cntにサンプル数が入ってる

                if len(null_values)>0:
                    data_points.append(null_value)
                    data_N.append(null_cnt)

                # X-RAYの計算する際は300,000行くらいでないと重くて時間がかかりすぎるのでサンプリング
                # データポイント、N数は各fold_modelで共通の為、初期データでのみ取得する
                self.point_dict[col] = data_points
                self.N_dict[col] = data_N

            if len(base_xray)>N_sample:
                self.df_xray = base_xray.sample(N_sample, random_state=fold_num)
            else:
                self.df_xray = base_xray
            #========================================================================
            # 一番計算が重くなる部分
            # multi_processにするとprocess数分のデータセットをメモリに乗せる必要が
            # あり, Overheadがめちゃでかく大量のメモリを食う。また、各データポイントに
            # ついてpredictを行うので、毎回全CPUを使っての予測が行われる。
            # また、modelオブジェクトが重いのか引数にmodelを載せて並列すると死ぬ
            #========================================================================
            # Multi threading
            if parallel:
                arg_list = []
                for point, N in zip(self.point_dict[col], self.N_dict[col]):
                    arg_list.append([col, point, N])

                #  for args in arg_list:
                    #  executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_point)
                #      executor.submit(self.parallel_xray_calculation, args)
                executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_point)
                futures = [executor.submit(self.parallel_xray_calculation, args) for args in arg_list]
                for future in concurrent.futures.as_completed(futures):
                    self.xray_list.append(future.result())
                executor.shutdown()

            else:
                for point, N in zip(self.point_dict[col], self.N_dict[col]):
                    one_xray = self.single_xray_calculation(col=col, val=point, N=N)
                    self.xray_list.append(one_xray)
            #========================================================================
            # 各featureのX-RAYの結果を統合
            #========================================================================
            tmp_result = pd.DataFrame(data=self.xray_list)

            self.xray_list = []
            gc.collect()

            if len(result):
                result = pd.concat([result, tmp_result], axis=0)
            else:
                result = tmp_result.copy()

        print(f"FOLD: {fold_num}")
        # 全てのfeatureのNとdata_pointを取得したら、全量データは必要なし
        try:
            del df_not_null
            del base_xray
        except UnboundLocalError:
            del base_xray
        gc.collect()

        return self, result
