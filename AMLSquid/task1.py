import pandas as pd
from sklearn.metrics import r2_score, make_scorer
from sklearn.linear_model import ElasticNet, HuberRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import KFold
from scipy.stats import zscore
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer
#TODO HUBER REGRESSOR
x_test = pd.read_csv('X_test.csv', header=0, index_col=0)
x_train = pd.read_csv('X_train.csv', header=0, index_col=0)
y_train = pd.read_csv('y_train.csv', header=0, index_col=0)


def outlier_detection(X):
    raise NotImplementedError


def feature_selection(model, X, y):
    raise NotImplementedError


def preprocessing(X):
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    zscores = pd.DataFrame()
    for feature in X.columns:
        imputer.fit(X[feature].values.reshape(-1, 1))
        new_feature = imputer.transform(X[feature].values.reshape(-1, 1)).flatten()
        zscores[feature] = zscore(new_feature)
        X[feature] = new_feature
    mean_zscore = zscores.mean(axis=1)

    return X, mean_zscore


def eNethyperParam(x_trainKF, x_valKF, y_trainKF, y_valKF):
    loss = []
    #alpha = [0.001, 0.002, 0.003, 0.01]
    epsilons = [1.3, 1.4, 1.6, 1.8, 4]
    alpha = [0.001, 0.0001, 0.001, 0.1]
    best_modelKF = ElasticNet()
    largest_score = -100
    for a in alpha:
        model = ElasticNet(alpha=a, fit_intercept=True, normalize=True, random_state=0, max_iter=10000)
        #model = HuberRegressor(epsilon=e, max_iter=10000, alpha=a)
        model.fit(x_trainKF, y_trainKF)
        y_hat = model.predict(x_valKF)
        score = r2_score(y_valKF, y_hat)
        loss.append(r2_score(y_valKF, y_hat))
        if largest_score < score:
            best_modelKF = model
            largest_score = score
    plt.plot(loss)
    plt.show()
    return best_modelKF, largest_score


# RandomForestClassifier
x_train, z_scores_train = preprocessing(x_train)
x_train['zscore'] = z_scores_train
x_train['y'] = y_train
indices = x_train[abs(x_train['zscore']) > 0.2].index
x_train.drop(indices, inplace=True)
y_train = x_train['y']
print(y_train.shape)
x_train.drop('y', axis=1, inplace=True)
x_train.drop('zscore', axis=1, inplace=True)
x_test, z_scores_test = preprocessing(x_test)


kf = KFold(shuffle=True)
kf.get_n_splits(x_train, y_train)

best_model = ElasticNet()
for train_i, val_i in kf.split(x_train, y_train):
    x_trainkf, X_val = x_train.values[train_i], x_train.values[val_i]
    y_trainkf, y_val = y_train.values[train_i], y_train.values[val_i]
    model, score = eNethyperParam(x_trainkf, X_val, y_trainkf.flatten(), y_val.flatten())
    best_model = model

print(score, best_model)
best_model.fit(x_train, y_train)
y_test = best_model.predict(x_test)
y_test_pd = pd.DataFrame(y_test, columns=['y'], dtype=int)
y_test_pd.index.name = 'id'
y_test_pd.to_csv("res.csv")
print(y_test_pd)
