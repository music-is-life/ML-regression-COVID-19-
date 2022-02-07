# importing libaries:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import dataset
dataset = pd.read_csv('COVID-19_Daily_Testing_-_By_Test.csv')
dataset = dataset.iloc[:,1:23] # removing also date column.
#checking data set:
print(dataset.isna().sum()) # no special NaN values.. only 1 row
dataset.dropna(inplace=True) # droping 1 row with NaN value of day...

X = dataset.drop(columns=['Not Positive Tests', 'Positive Tests'], inplace=False)
y = dataset.iloc[:,1:3]

# DF for summery table:
final_table = pd.DataFrame(columns=['Model', 'best_Parameters', 'R2_CV_score'])
final_table['Model'] = ['Linear regression', 'Polynomial',
                            'Ridge regression', 'Lasso', 'Random forest',
                            'Knn']
### Preaparing data:

#encoding:
from sklearn.preprocessing import OneHotEncoder,LabelEncoder;
from sklearn.compose import ColumnTransformer;
labelencoder_X = LabelEncoder()
X.iloc[:,0] = labelencoder_X.fit_transform(X.iloc[:,0])

# # checking for multicollinearity 
import seaborn as sns
# after a check Total tests, female and male tests seems to have multicolinirity.    
# the scores of the models even got better.
X = X.drop(columns=['Total Tests', 'Tests - Male', 'Tests - Female', 'Tests - Age 30-39', 'Tests - Age 50-59'], inplace=False)

# heat map
matrix = np.triu(X.corr(), 1)
plt.figure(figsize = (20, 20))
sns.heatmap(X.corr(), annot = True, square = True, mask = matrix, linewidths = 0.5, annot_kws = {'size': 20})
plt.title('Correlation Matrix', fontsize = 40)
plt.tick_params(labelsize = 20)
plt.show()

ct = ColumnTransformer([("Day", OneHotEncoder(), [0])], remainder = 'passthrough');
X = ct.fit_transform(X) # continue encoding...

# scaling:
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X = sc_x.fit_transform(X)

# splitting data to train and test sets:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

### Linear regression:
from sklearn.linear_model import LinearRegression, Ridge, Lasso
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)

#cross validation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
accuracies = cross_val_score(estimator=lin_reg, X=X_test, y=y_test, cv=cv, scoring='r2', n_jobs = 1)
final_table.iloc[0,2] = accuracies.mean()
print("linear regression")
print(f"mean of r2: {accuracies.mean()}")
print(f"mean of std r2: {accuracies.std()}\n")

### Polynomial regression:
from sklearn.preprocessing import PolynomialFeatures
# degree 2
poly_reg = PolynomialFeatures (degree = 2)
X_poly = poly_reg.fit_transform(X_train)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y_train)
y_poly_pred = lin_reg_2.predict(poly_reg.fit_transform(X_test))

# cross validation and evaluation for degree level:
from sklearn.metrics import mean_squared_error, r2_score
rmse = np.sqrt(mean_squared_error(y_test,y_poly_pred))
r2 = r2_score(y_test, y_poly_pred)

final_table.iloc[1,2] = r2
final_table.iloc[1,1] = 'degree = 2'

print("Polyminal regression degree 2: ")
print(f"mean of r2: {r2}")
print(f"rmse is: {rmse}\n")

# degree 3
poly_reg = PolynomialFeatures (degree = 3)
X_poly = poly_reg.fit_transform(X_train)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y_train)
y_poly_pred = lin_reg_2.predict(poly_reg.fit_transform(X_test))

# cross validation and evaluation for degree level:
rmse = np.sqrt(mean_squared_error(y_test,y_poly_pred))
r2 = r2_score(y_test, y_poly_pred)

print("Polyminal regression degree 3: ")
print(f"mean of r2: {r2}")
print(f"rmse is: {rmse}\n")
# degree 4
poly_reg = PolynomialFeatures (degree = 4)
X_poly = poly_reg.fit_transform(X_train)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y_train)
y_poly_pred = lin_reg_2.predict(poly_reg.fit_transform(X_test))

# cross validation and evaluation for degree level:
rmse = np.sqrt(mean_squared_error(y_test,y_poly_pred))
r2 = r2_score(y_test, y_poly_pred)

print("Polyminal regression degree 4: ")
print(f"mean of r2: {r2}")
print(f"rmse is: {rmse}\n") # best polynom degree is 3.

### Ridge regression:
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

from sklearn.linear_model import RidgeCV, MultiTaskLassoCV
model = RidgeCV(alphas=np.linspace(0.01, 5.0, num=100), cv=3)
# fit model
model.fit(X_train, y_train)

accuracies = cross_val_score(estimator=model, X=X_test, y=y_test, cv=10, scoring='r2', n_jobs = -1)
print("Ridge regression")
print(f"best alpha is: {model.alpha_}") # best alpha for ridge
print(f"mean of r2: {accuracies.mean()}")
final_table.iloc[2,2] = accuracies.mean()
final_table.iloc[2,1] = f"best alpha is: {model.alpha_}\n"

### Lasso regression:
model = MultiTaskLassoCV(alphas=np.linspace(0.01, 5.0, num=100), cv=3, n_jobs = -1)
# fit model
model.fit(X_train, y_train)

accuracies = cross_val_score(estimator=model, X=X_test, y=y_test, cv=10, scoring='r2', n_jobs = -1)
print("Lasso regression")
print(f"best alpha is: {model.alpha_}") # best alpha for ridge
print(f"mean of r2: {accuracies.mean()}")
final_table.iloc[3,2] = accuracies.mean()
final_table.iloc[3,1] = f"best alpha is: {model.alpha_}"


# Compute the cross-validation score with the default hyper-parameters
from sklearn.model_selection import cross_val_score
for Model in [Ridge, Lasso]:
    model = Model()
    print('%s: %s' % (Model.__name__,
                      cross_val_score(model, X_train, y_train).mean()))

# We compute the cross-validation score as a function of alpha, the strength of the regularization for Lasso and Ridge


alphas = np.linspace(0.01, 5.0, num=100)

plt.figure(figsize=(5, 3))

for Model in [Lasso, Ridge]:
    scores = [cross_val_score(Model(alpha), X_test, y_test, cv=3, n_jobs = -1).mean()
            for alpha in alphas]
    plt.plot(alphas, scores, label=Model.__name__)

plt.legend(loc='lower left')
plt.xlabel('alpha')
plt.ylabel('cross validation score')
plt.tight_layout()
plt.show()
print(max(scores))

### Random forest:
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
rf_Model = RandomForestClassifier()
regr_multirf = RandomForestRegressor(n_estimators=10, random_state=0)
regr_multirf.fit(X_train, y_train)

param_grid = {'n_estimators': [int(x) for x in np.linspace(start = 10, stop = 80, num = 10)],
                'max_features': ['auto', 'sqrt'],
                'max_depth': [2,4],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'bootstrap': [True, False]}

from sklearn.model_selection import RandomizedSearchCV
rf_RandomGrid = RandomizedSearchCV(estimator = regr_multirf, param_distributions = param_grid, cv = 3, verbose=2, n_jobs = -1)
rf_RandomGrid.fit(X_train, y_train)

accuracies = cross_val_score(estimator=rf_RandomGrid, X=X_test, y=y_test, cv=cv, scoring='r2', n_jobs = -1)
print("Random Forest: ")
print(f"The best parameters:\n {rf_RandomGrid.best_params_}")
print(f"mean of r2: {accuracies.mean()}")

final_table.iloc[4,2] = accuracies.mean()
final_table.iloc[4,1] = rf_RandomGrid.best_params_['n_estimators']

### Knn Regressor:
from sklearn.neighbors import KNeighborsRegressor

algorithm = KNeighborsRegressor()   

hp_candidates = [{'n_neighbors': [2,3,4,5,6], 'weights': ['uniform','distance']}]
knn_model = RandomizedSearchCV(estimator=algorithm, param_distributions=hp_candidates, cv=cv, scoring='r2')
knn_model.fit(X, y)

accuracies = cross_val_score(estimator=knn_model, X=X_test, y=y_test, cv=cv, scoring='r2', n_jobs = -1)
print("KNeighbors regression: ")
print(f"The best parameters:\n {knn_model.best_params_}")
print(f"mean of r2: {accuracies.mean()}")

final_table.iloc[5,2] = accuracies.mean()
final_table.iloc[5,1] = knn_model.best_params_['n_neighbors']

### FINAL SUMMARY of all models and what i choose for the data prediction:

print(final_table)