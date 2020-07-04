import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score,mean_absolute_error,mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
url='C:\\Users\\KARTHIK\\PycharmProjects\\carprediticion\\car data.csv'
da = pd.read_csv(url)

finalDataSet = da.drop(["Car_Name"],axis=1)

finalDataSet["no_of_Year"]=2020-finalDataSet["Year"]

finalDataSet.drop(["Year"],axis=1,inplace=True)

finalDataSet = pd.get_dummies(finalDataSet,drop_first=True)
x = finalDataSet.loc[:,"Present_Price":]
y = finalDataSet.loc[:,"Selling_Price"]


model = ExtraTreesRegressor()
model.fit(x,y)
print(model.feature_importances_)
top = pd.Series(data=model.feature_importances_,index=x.columns)
top = top.sort_values(ascending=False).head(4)

TrainX,TestX,TrainY,TestY=train_test_split(x,y,test_size=0.2,random_state=0)
rf = RandomForestRegressor()

#Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]
params=\
    {'n_estimators':n_estimators,
    'max_features':max_features,
    'max_depth':max_depth,
    'min_samples_split':min_samples_split,
    'min_samples_leaf':min_samples_leaf}

rf_random=RandomizedSearchCV(estimator=rf,param_distributions=params,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)
rf_random.fit(TrainX,TrainY)
# print(rf_random.best_estimator_)
# print(rf_random)
pred = rf_random.predict(TestX)


#Accuracy
print('MAE:',mean_absolute_error(TestY, pred))
print('MSE:', mean_squared_error(TestY, pred))
print('RMSE:', np.sqrt(mean_squared_error(TestY, pred)))
file = open('random_forest_regression_model.pkl', 'wb')

# dump information to that file
pickle.dump(rf_random, file)