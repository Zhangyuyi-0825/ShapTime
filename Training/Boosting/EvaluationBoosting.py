import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

def accuracy(model, x_train, y_train, x_test, y_test):
    model_eva = model.fit(x_train, y_train)
    
    pre = model_eva.predict(x_test)
    r2 = r2_score(pre, y_test)
    mse = mean_squared_error(pre, y_test)  
    
    return r2, mse


def feature(data, target):
    x_eva = data.drop(columns = {target}, axis = 1)
    y_eva = data.loc[:, [target]]
    
    return x_eva, y_eva


def generate(set_eva, guide_list):  #len(guide_list) = len(set_eva)
    df = set_eva[ guide_list[0] ]
    for i in range(1, len(guide_list)):
        df = pd.concat([ df, set_eva[ guide_list[i] ] ], axis = 0)
        
    return df

    
def evaluation(set_eva, x_test, y_test, target, model, guide_list): # set_eva = df_eva
    r2_results = []
    mse_results = []
    
    for i in range(len(guide_list)):
        df = generate(set_eva, guide_list[i])
        x_eva, y_eva = feature(df, target)
        r2, mse = accuracy(model, x_eva, y_eva, x_test, y_test)
        
        r2_results.append(r2)
        mse_results.append(mse)
        
    return r2_results, mse_results


def R2Plot(r2_results, title_name):
    sns.reset_orig()
    plt.figure(figsize = (5,4))
    plt.title(title_name, fontsize = 24)
    plt.plot(r2_results, label = 'r2')

    plt.legend(fontsize = 20)
    plt.show()

def MSEPlot(mse_results, title_name):
    sns.reset_orig()
    plt.figure(figsize = (5,4))
    plt.title(title_name, fontsize = 24)
    plt.plot(mse_results, label = 'MSE')

    plt.legend(fontsize = 20)
    plt.show()
