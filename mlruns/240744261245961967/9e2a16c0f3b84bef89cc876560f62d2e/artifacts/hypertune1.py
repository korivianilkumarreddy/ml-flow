from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import pandas as pd
import mlflow

#Load the breast Cancer Dataset
data = load_breast_cancer()
x = pd.DataFrame(data.data,columns = data.feature_names)
y = pd.Series(data.target,name = 'target')
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)
rf = RandomForestClassifier(random_state=42)
#defining the parameter grid for grid search cv
param_grid = {
    'n_estimators' : [10,50,100],
    'max_depth' : [None,10,20,30]
}
#Applying Grid search cv
grid_search = GridSearchCV(estimator = rf,param_grid = param_grid,cv =5,n_jobs = -1,verbose = 2)
'''#run without MLflow from here
grid_search.fit(x_train,y_train)

#Displaying the best params and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(best_params)
print(best_score)
#Till here'''
mlflow.set_experiment('breast-cancer-rf-hp')

with mlflow.start_run():
    grid_search.fit(x_train,y_train)

    #displaying the parametrs and the best score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    #log params
    mlflow.log_params(best_params)

    #log metrics
    mlflow.log_metric("accuracy",best_score)

    #log training data
    train_df = x_train.copy()
    train_df['target'] = y_train
    train_df = mlflow.data.from_pandas(train_df)
    mlflow.log_input(train_df,"training")

    #log test data
    test_df = x_test.copy()
    test_df['target'] = y_test
    test_df = mlflow.data.from_pandas(test_df)
    mlflow.log_input(test_df,"testing")

    #log source code
    mlflow.log_artifact(__file__)
    
    #log the best model
    mlflow.sklearn.log_model(grid_search.best_estimator_,"random_forest")

    #set tags
    mlflow.set_tag("author","anil kumar")

    print(best_params)
    print(best_score)