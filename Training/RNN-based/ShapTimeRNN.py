import numpy as np
import pandas as pd
import math

import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Dropout, Bidirectional


# create super-time
def supertime(Tn, data):
    
    dfx = []
    lenth = int(len(data)/Tn)
    start = len(data) - (lenth * Tn)
    
    data_s = data[start:, :, :]
    
    for i in range(Tn):
        Ti = data_s[ i*lenth : (i+1)*lenth, :, : ]
        dfx.append(Ti)
        
    return dfx

def supertime_add(Tn, data):
    
    dfx = []
    lenth = int(len(data)/Tn)
    start = len(data) - (lenth * Tn)
    
    data_s = data.iloc[start: , :]
    
    for i in range(Tn):
        Ti = data_s.iloc[ i*lenth : (i+1)*lenth ]
        dfx.append(Ti)
        
    return dfx


# create ShapTime
def get_sub_set(Tn):
    
    mylist = list(range(Tn))
    sub_sets = [[]]
    for x in mylist:
        sub_sets.extend([item + [x] for item in sub_sets])
    return sub_sets 


def ValFunction(model, interp_x, Tn):
    
    dfx = supertime(Tn, interp_x)
    # Generate subsets of indexes
    subset = get_sub_set(Tn)   
    
    # Generate the baseline
    y_results = model.predict(interp_x)
    baseline = sum(y_results)/len(interp_x)
    
    val_results = []
    
    for i in range(1, len(subset)):
        x_i = dfx[subset[i][0]]
        
        if len(subset[i]) == 1:
            prediction = model.predict(x_i)
            results = (sum(prediction)/len(x_i)) - baseline
            val_results.append(results)
            
        else:
            for n in range(1, len(subset[i])):
                x_i = np.vstack([x_i, dfx[subset[i][n]]])
            
            prediction = model.predict(x_i)
            results = (sum(prediction)/len(x_i)) - baseline
            val_results.append(results)
            
    val_results.insert(0,0.0)
            
    return subset, val_results


def index(Si, subset):
    for i in range(len(subset)):
        if Si == subset[i]:
            index = i
        else:
            pass
    return index


def ShapleyValues(model, interp_x, Tn):
    
    subset, val_results = ValFunction(model, interp_x, Tn)
    shapley_values = []
    for i in range (Tn):
        shapley = []
        for n in range(len(subset)):
            if i not in subset[n]:
                Si = subset[n]+[i]
                Si.sort()
                Si_num = index(Si, subset)
            
                S_num = len(subset[n])
                N = Tn
            
                weight = (math.factorial(S_num) * math.factorial((N-S_num-1))) / math.factorial(N)
                val = val_results[Si_num] - val_results[n]
                shapley_i = weight * val
            
                shapley.append(shapley_i)
            else:
                pass
            
        shapley_values.append(sum(shapley))
        del shapley
        
    return shapley_values


def trans(original):
    results_exp = []
    for i in range(len(original)):
        results_exp.append(float(original[i]))
    return results_exp


def TimeImportance(Tn, ST_value, time_columns):
    time_list = list(range(Tn))
    shapley_impor = pd.DataFrame(index = time_list, columns = ['ShapTime'])
    shapley_impor['ShapTime'] = ST_value
    shapley_impor['absolute'] = abs(shapley_impor['ShapTime'])
    
    shapley_impor.index = time_columns
    shapley_impor.sort_values(by='absolute', inplace = True, ascending=False)
    
    sns.set(context='paper', style='ticks', font_scale=2)
    ax = sns.barplot(x="ShapTime", y=shapley_impor.index, data=shapley_impor, orient = 'h', color="lightskyblue", palette = 'Blues_r')
    plt.show()
    
    
def TimeHeatmap(Tn, ST_value, time_columns):
    time_list = list(range(Tn))
    shapley_df = pd.DataFrame(index = time_list, columns = ['ShapTime'])
    shapley_df['ShapTime'] = ST_value
    shapley_df_abs = abs(shapley_df)
    
    shapley_df_abs.index = time_columns
    sns.set(font_scale=1.3)
    f, ax = plt.subplots(figsize=(13, 1))
    sns.heatmap(shapley_df_abs.T, annot=False, linewidths=0, ax=ax, cmap = 'Blues')






