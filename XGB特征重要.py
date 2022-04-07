# plot feature importance using built-in function
from numpy import loadtxt
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier
from xgboost import plot_importance

import matplotlib # 注意这个也要import一次
import matplotlib.pyplot as plt
import pandas as pd
# load data
data = pd.read_excel(r'E:\LJ\2d.xlsx', sheet_name='Sheet2')
# split data into X and y

X = data.iloc[:,2:13]
y = data['lei']
# fit model no training data
model = XGBClassifier()
#model= RandomForestRegressor()
model.fit(X, y)
# plot feature importance
plot_importance(model)
#plt.gcf().subplots_adjust(left=0.5,top=0.91,bottom=0.09, right=None)


#pyplot.show().subplots_adjust(left=None,top=None,bottom=None, right=None)

plt.show()