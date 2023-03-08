# ShapTime
A paper that has been submitted to IntelliSys2023.
The core of the work is to build the general explainable method specifically for time series forecasting.
The focus is on explaining the importance of time itself, so that the forecasting performance can be improved based on that.

Boosting and RNN models are used in time series forecasting tasks

[Explanation for Boosting](https://github.com/Zhangyuyi-0825/ShapTime/blob/master/Training/Boosting/ShapTimeBoosting.py)

[Explanation for RNN-based](https://github.com/Zhangyuyi-0825/ShapTime/blob/master/Training/RNN-based/ShapTimeRNN.py)

Visualization of Explanation Results:
![image](https://github.com/Zhangyuyi-0825/ShapTime/blob/master/image/explanation.png)

The darker the color, the more important the time period. This means that the forecasting model more focus on this time period during the training process.
After that, use the information presented by these explanation results as the guide to retrain the original forecasting model and try to improve the forecasting performance.

[Improvement.ipynb](https://github.com/Zhangyuyi-0825/ShapTime/blob/master/Improvement.ipynb)

