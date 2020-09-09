# Benchmark LightGBM to compare changes in hardware, compiler settings, and software versions
# By Andrew Ziem
# Copyright (C) 2020 by Compassion International, Inc.

# References:
#  http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html
#  https://github.com/timeamagyar/kdd-cup-99-python/blob/master/kdd%20preprocessing.ipynb

# imports
import lightgbm as lg
import time
import pandas as pd

# constants
data_url = 'http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz'
#data_url = 'http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz'
num_boost_round = 5000
lg_params = {
    'objective' :'binary',
    'learning_rate' : 0.01,
    'num_leaves' : 32,
    'feature_fraction': 0.5, 
    'bagging_fraction': 0.5, 
    'bagging_freq' : 1,
    'boosting_type' : 'gbdt',
    'metric': 'binary_logloss'
}
names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]

# read data
def get_data():
    """Return a LightGBM data set with the KDD 1999 data"""
    print(f'Reading data from {data_url}')
    df = pd.read_csv(data_url, names=names)
    print(f' row count: {df.shape[0]:,}')
    print(f' original column count: {df.shape[1]}')

    print('Removing invariant features')
    df['num_outbound_cmds'].value_counts()
    df.drop('num_outbound_cmds', axis=1, inplace=True)
    df['is_host_login'].value_counts()
    df.drop('is_host_login', axis=1, inplace=True)
    print(f' new column count: {df.shape[1]}')

    print('Transforming categorical variables')
    df['protocol_type'] = df['protocol_type'].astype('category')
    df['service'] = df['service'].astype('category')
    df['flag'] = df['flag'].astype('category')
    cat_columns = df.select_dtypes(['category']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)

    print('Collapsing values in label column')
    df['label'] = (df['label']=='normal.').astype(int)

    print('Converting to LightGBM data set format')
    X_train = df.drop('label', axis=1)
    Y_train = df['label']
    d_train = lg.Dataset(X_train, Y_train)
    return d_train


def classify(d_train):
    print(f'Training for {num_boost_round} boosting iterations')
    time_start = time.time()
    model = lg.train(lg_params, d_train, num_boost_round=num_boost_round)
    elapsed_seconds = time.time() - time_start
    print(f' elapsed training time: {elapsed_seconds:.0f} seconds ({elapsed_seconds/60.0:.2f} minutes)')

def show_sysinfo():
    """Show some basic information about the system"""
    print('Showing system information')
    import socket
    print(f' hostname: {socket.getfqdn()}')
    import platform
    print(f' processor: {platform.processor()}')
    import os
    print(f' cpu count: {os.cpu_count()}')
    try:
        import cpuinfo
        cpu = cpuinfo.get_cpu_info()
        print(f' cpu brand: {cpu["brand_raw"]}')
        print(f' cpu hz advertised: {cpu["hz_advertised_friendly"]}')
        print(f' cpu hz actual: {cpu["hz_actual_friendly"]}')
    except:
        print(' no cpuinfo, try "pip install py-cpuinfo"')

def go():
    show_sysinfo()
    classify(get_data())

if __name__ == '__main__':
    go()
