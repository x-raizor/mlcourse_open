
# coding: utf-8

import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

from matplotlib import pyplot as plt
import seaborn as sns

# загрузим обучающую и тестовую выборки
train_df = pd.read_csv('../../data/alice/train_sessions.csv',
                       index_col='session_id')
test_df = pd.read_csv('../../data/alice/test_sessions.csv',
                      index_col='session_id')

# приведем колонки time1, ..., time10 к временному формату
times = ['time%s' % i for i in range(1, 11)]
train_df[times] = train_df[times].apply(pd.to_datetime)
test_df[times] = test_df[times].apply(pd.to_datetime)

# отсортируем данные по времени
train_df = train_df.sort_values(by='time1')


# приведем колонки site1, ..., site10 к целочисленному формату и заменим пропуски нулями
sites = ['site%s' % i for i in range(1, 11)]
train_df[sites] = train_df[sites].fillna(0).astype('int')
test_df[sites] = test_df[sites].fillna(0).astype('int')

# загрузим словарик сайтов
with open(r"../../data/alice/site_dic.pkl", "rb") as input_file:
    site_dict = pickle.load(input_file)

# датафрейм словарика сайтов
sites_dict_df = pd.DataFrame(list(site_dict.keys()), 
                          index=list(site_dict.values()), 
                          columns=['site'])

# наша целевая переменная
y_train = train_df['target']

# объединенная таблица исходных данных
full_df = pd.concat([train_df.drop('target', axis=1), test_df])

# индекс, по которому будем отделять обучающую выборку от тестовой
idx_split = train_df.shape[0]

sites = ['site%d' % i for i in range(1, 11)]
# табличка с индексами посещенных сайтов в сессии
full_sites = full_df[sites]

# последовательность с индексами
sites_flatten = full_sites.values.flatten()

# искомая матрица
full_sites_sparse = csr_matrix(([1] * sites_flatten.shape[0],
                                sites_flatten,
                                range(0, sites_flatten.shape[0] + 10, 10)))[:, 1:]

X_train_sparse = full_sites_sparse[:idx_split]
X_test_sparse = full_sites_sparse[idx_split:]


def get_auc_lr_valid(X, y, C=1.0, ratio = 0.9, seed=17):
    '''
    X, y – выборка
    ratio – в каком отношении поделить выборку
    C, seed – коэф-т регуляризации и random_state 
              логистической регрессии
    '''
    train_len = int(ratio * X.shape[0])
    X_train = X[:train_len, :]
    X_valid = X[train_len:, :]
    y_train = y[:train_len]
    y_valid = y[train_len:]
    
    logit = LogisticRegression(C=C, n_jobs=-1, random_state=seed)
    
    logit.fit(X_train, y_train)
    
    valid_pred = logit.predict_proba(X_valid)[:, 1]
    
    return roc_auc_score(y_valid, valid_pred)



# функция для записи прогнозов в файл
def write_to_submission_file(predicted_labels, out_file,
                             target='target', index_label="session_id"):
    predicted_df = pd.DataFrame(predicted_labels,
                                index = np.arange(1, predicted_labels.shape[0] + 1),
                                columns=[target])
    predicted_df.to_csv(out_file, index_label=index_label)


# **Обучите модель на всей выборке, сделайте прогноз для тестовой выборки и сделайте посылку в соревновании**.


new_feat_train = pd.DataFrame(index=train_df.index)
new_feat_test = pd.DataFrame(index=test_df.index)

new_feat_train['year_month'] = train_df['time1'].apply(lambda ts: 100 * ts.year + ts.month)
new_feat_test['year_month'] = test_df['time1'].apply(lambda ts: 100 * ts.year + ts.month)
new_feat_train['months'] = train_df['time1'].apply(lambda ts: 12 * ts.year + ts.month )
new_feat_test['months'] = test_df['time1'].apply(lambda ts: 12 * ts.year + ts.month )

scaler = StandardScaler()
scaler.fit(new_feat_train['months'].values.reshape(-1, 1))

new_feat_train['months_scaled'] = scaler.transform(new_feat_train['months'].values.reshape(-1, 1))
new_feat_test['months_scaled'] = scaler.transform(new_feat_test['months'].values.reshape(-1, 1))


X_train_sparse_new = csr_matrix(hstack([X_train_sparse, 
                             new_feat_train['months_scaled'].values.reshape(-1, 1)]))


import math

new_feat_train['hour_cos'] = train_df['time1'].apply(lambda ts: math.cos(ts.hour * math.pi / 12))
new_feat_train['hour_sin'] = train_df['time1'].apply(lambda ts: math.sin(ts.hour * math.pi / 12))
new_feat_test['hour_cos'] = test_df['time1'].apply(lambda ts: math.cos(ts.hour * math.pi / 12))
new_feat_test['hour_sin'] = test_df['time1'].apply(lambda ts: math.sin(ts.hour * math.pi / 12))


new_feat_train['alice_time_chance'] = train_df['time1'].apply(lambda ts: 1 \
  if ((ts.hour > 8.5) and (ts.hour < 9.5)) or  \
  ((ts.hour > 11.5) and (ts.hour < 13.5)) or \
  ((ts.hour > 14.5) and (ts.hour < 18.5))  else 0)

new_feat_test['alice_time_chance'] = test_df['time1'].apply(lambda ts: 1 \
  if ((ts.hour > 8.5) and (ts.hour < 9.5)) or  \
  ((ts.hour > 11.5) and (ts.hour < 13.5)) or \
  ((ts.hour > 14.5) and (ts.hour < 18.5))  else 0)


scaler = StandardScaler()
scaler.fit(new_feat_train[['hour_cos', 'hour_sin', 'alice_time_chance']])

tre_matrix = scaler.transform(new_feat_train[['hour_cos', 'hour_sin', 'alice_time_chance']])
tre_matrix_t = scaler.transform(new_feat_test[['hour_cos', 'hour_sin', 'alice_time_chance']])

new_feat_train['hour_cos_scaled'] = tre_matrix[:,0]
new_feat_train['hour_sin_scaled'] = tre_matrix[:,1]
new_feat_train['alice_time_chance'] = tre_matrix[:,2]

new_feat_test['hour_cos_scaled'] = tre_matrix_t[:,0]
new_feat_test['hour_sin_scaled'] = tre_matrix_t[:,1]
new_feat_test['alice_time_chance'] = tre_matrix_t[:,2]


X_train_sparse_new = csr_matrix(hstack([
    X_train_sparse, 
    new_feat_train['alice_time_chance'].values.reshape(-1, 1),
    new_feat_train['hour_sin_scaled'].values.reshape(-1, 1),
    new_feat_train['hour_cos_scaled'].values.reshape(-1, 1)]))


X_test_sparse_new = csr_matrix(hstack([
    X_test_sparse, 
    new_feat_test['alice_time_chance'].values.reshape(-1, 1),
    new_feat_test['hour_sin_scaled'].values.reshape(-1, 1),
    new_feat_test['hour_cos_scaled'].values.reshape(-1, 1)]))



logit = LogisticRegression(C=0.5, n_jobs=-1, random_state=17)
nlogit.fit(X_train_sparse_new, y_train)


test_pred = logit.predict_proba(X_test_sparse_new)[:, 1]

write_to_submission_file(test_pred, 'alice-two-timevariables-regularization.csv')

