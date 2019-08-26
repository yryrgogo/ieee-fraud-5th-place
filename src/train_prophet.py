import gc
import datetime as dt 
import numpy as np
import pandas as pd
import seaborn as sns
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
sns.set(style='whitegrid')


def train_prophet(model, train, test, COLUMN_TARGET, ADD_FEATURES=[], params={}, is_log=False, is_contract=True, pred_period=366, is_viz=False):
    
    if is_contract:
        z_thres = 3.0
        avg = train[COLUMN_TARGET].mean()
        std = train[COLUMN_TARGET].std()
        train['z'] = (train.loc[:, COLUMN_TARGET] - avg) / std
        const_max = train[train['z'] <=    z_thres][COLUMN_TARGET].max()
        const_min = train[train['z'] >= -1*z_thres][COLUMN_TARGET].min()
        train.loc[train['z'] >    z_thres, COLUMN_TARGET] = const_max
        train.loc[train['z'] < -1*z_thres, COLUMN_TARGET] = const_min
        train.drop('z', axis=1, inplace=True)
    
    if is_log:
        train[COLUMN_TARGET] = train[COLUMN_TARGET].map(lambda x: np.log1p(x))

    if params['growth']=='logistic':
        train['cap'] = 13.0
        
    train.rename(columns={'date':'ds', COLUMN_TARGET:'y'}, inplace=True)
    model.fit(train)
    
    future = model.make_future_dataframe(periods=pred_period)
    
    if len(test):
        test.rename(columns={'date':'ds', COLUMN_TARGET:'y'}, inplace=True)
        all_data = pd.concat([train, test], axis=0, ignore_index=True)
    else:
        all_data = train
    
    if len(ADD_FEATURES):
        future.set_index('ds', inplace=True)
        all_data.set_index('ds', inplace=True)
        future = future.join(all_data[ADD_FEATURES], how='left')
        future.reset_index(inplace=True)
        future.rename(columns={'index':'ds'}, inplace=True)
        
    if params['growth']=='logistic':
        future['cap'] = 13.0
    forecast = model.predict(future)
    
    if is_log:
        no_pred_columns = ['ds', 'cap']
        pred_columns = [col for col in forecast if col not in no_pred_columns]
        forecast[pred_columns] =  np.expm1(forecast[pred_columns].values)
    
    if is_viz:
        fig = model.plot(forecast)
        if len(ADD_FEATURES):
            model.plot_components(forecast)
        
        add_changepoints_to_plot(fig.gca(), model, forecast)
        
        plt.show()
    
    return forecast, model


def plot_prophet(all_data, forecast, COLUMN_TARGET):
    
    forecast['ds'] = forecast['ds'].map(lambda x: x.date())
    
    viz_columns = ['ds', 'target']
    tmp_org     = all_data[['date', COLUMN_TARGET]]
    tmp_trend   = forecast[['ds', 'trend']]
    tmp_yhat    = forecast[['ds', 'yhat']]
        
    tmp_org.columns     = viz_columns
    tmp_trend.columns   = viz_columns
    tmp_yhat.columns    = viz_columns
    tmp_org['category']     = 'observed'
    tmp_trend['category']   = 'trend'
    tmp_yhat['category']    = 'predicted'
    concat_list = [tmp_org, tmp_trend, tmp_yhat]
    
    if 'yearly' in forecast.columns:
        tmp_yearly  = forecast[['ds', 'yearly']]
        tmp_yearly.columns  = viz_columns
        tmp_yearly['category']  = 'yearly'
        concat_list.append(tmp_yearly)
    if 'monthly' in forecast.columns:
        tmp_monthly  = forecast[['ds', 'monthly']]
        tmp_monthly.columns  = viz_columns
        tmp_monthly['category']  = 'monthly'
        concat_list.append(tmp_monthly)
    if 'weekly' in forecast.columns:
        tmp_weekly  = forecast[['ds', 'weekly']]
        tmp_weekly.columns  = viz_columns
        tmp_weekly['category']  = 'weekly'
        concat_list.append(tmp_weekly)
    
    df_viz = pd.concat(concat_list, axis=0)
    df_viz['target'] = df_viz['target'].astype('int')
    
    print(f"Time Series Element")
    plt.figure(figsize=(30, 8))
    sns.set(style='whitegrid')
    sns.lineplot(data=df_viz, x='ds', y='target', hue='category')
    plt.show()
    
    print(f"Residuals")
    plt.figure(figsize=(30, 4))
    forecast['residual'] = forecast[COLUMN_TARGET] - forecast['yhat']
    sns.lineplot(data=forecast, x='ds', y='residual')
    plt.show()
    
    return forecast

    
def scoring_prophet(forecast, COLUMN_TARGET, trend_term):
        
    pred_term = forecast.copy()
    
    if trend_term=='yearly':
        pattern_list = sorted(pred_term['month'].unique())
        column_pattern = 'month'
    elif trend_term=='monthly':
        pattern_list = sorted(pred_term['day'].unique().tolist())
        column_pattern = 'day'
    elif trend_term=='weekly':
        pattern_list = sorted(pred_term['day_of_week'].unique().tolist())
        column_pattern = 'day_of_week'
    score_result = {}
    
    def scoring(pred_tmp, result):
        r2   = r2_score(pred_tmp[COLUMN_TARGET].values, pred_tmp['yhat'].values)
        mape = np.abs(pred_tmp['residual'].values / pred_tmp[COLUMN_TARGET].values).mean()
        rmse = np.sqrt(mean_squared_error(pred_tmp[COLUMN_TARGET].values, pred_tmp['yhat'].values))
        print(f"{trend_term}/{pattern} | R2: {r2} | MAPE: {mape} | RMSE: {rmse}")
        
        result['trend_term'] = trend_term
        result[column_pattern] = pattern
        result['r2'] = r2
        result['mape'] = mape
        result['rmse'] = rmse
        return result
    
    print(pattern_list)
    for pattern in pattern_list:
        pred_tmp = pred_term[pred_term[column_pattern]==pattern]
        score_result = scoring(pred_tmp, score_result)
    else:
        pred_tmp = pred_term
        pattern = 'all'
        score_result = scoring(pred_tmp, score_result)
        
    return score_result


#========================================================================
# Train
#========================================================================

def main_prophet(train, test, COLUMN_TARGET, ADD_FEATURES, pred_period=366, trend_term='yearly', is_viz=False, is_score=False, params={}, is_contract=True, holidays=[]):
    
    # Parameter
    if len(params)==0:
        params = {
            'growth': ['linear', 'logistic'][0],
            'yearly_seasonality': 'auto',
            'weekly_seasonality': True,
            'daily_seasonality': False ,
            'holidays_prior_scale': 10,
            'changepoint_range': 1.0,
            'changepoint_prior_scale': 0.05,
        #     'mcmc_samples': 1000,
        }

    # For Result Save
    pred_list = []
    residual_list = []
    
    
    # Base Columns
    use_cols = ['date', COLUMN_TARGET]
    
    if len(holidays):
        model = Prophet(**params, holidays=holidays)
    else:
        model = Prophet(**params)
    if len(ADD_FEATURES):
        for feature in ADD_FEATURES:
            model.add_regressor(feature)
            use_cols.append(feature)
    # model.add_seasonality(name='monthly', period=monthly_period, fourier_order=5)
    
    if len(test):
        tmp_test = test[use_cols]
    else:
        tmp_test = test

    # Train & Prediction
    forecast, model = train_prophet(model, train[use_cols], tmp_test, COLUMN_TARGET, ADD_FEATURES, params, is_contract=is_contract, pred_period=pred_period, is_viz=is_viz)
    
    del tmp_test
    gc.collect()
    
    if len(test):
        all_data = pd.concat([train, test], axis=0, ignore_index=False)
    else:
        all_data = train
    
    all_data.set_index('date', inplace=True)
    forecast['year'] = forecast['ds'].map(lambda x: x.year)
    forecast['month'] = forecast['ds'].map(lambda x: x.month)
    forecast['day'] = forecast['ds'].map(lambda x: x.day)
    forecast.set_index('ds', inplace=True)
    forecast['day_of_week'] = all_data['day_of_week']
    forecast[COLUMN_TARGET] = all_data[COLUMN_TARGET]
    forecast = forecast[~forecast[COLUMN_TARGET].isnull()]
    forecast[COLUMN_TARGET] = forecast[COLUMN_TARGET].astype('int')
    forecast.reset_index(inplace=True)
    all_data.reset_index(inplace=True)
    
#     print("  * Prophet Train & Prediction Done!")

    # Viz
    if is_viz:
        forecast = plot_prophet(all_data, forecast, COLUMN_TARGET)
#         print("  * Get Visualize Source Done!")
    
    # Evaluation
    if is_score:
        score_result = scoring_prophet(forecast, COLUMN_TARGET, trend_term)
#         print("  * Scoring Done!")

    # Result
    forecast_columns = [col for col in forecast if not col.count('additive') and not col.count('multiplicative')]
    df_pred = forecast[forecast_columns]
    df_pred.rename(columns={COLUMN_TARGET: 'observed'}, inplace=True)
    
#     print("* Prophet All Process Done!")

    return df_pred