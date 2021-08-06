# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 16:14:14 2021

@author: foster-s
"""

import numpy  as np
import pandas as pd
import warnings
from AdvancedAnalytics.ReplaceImputeEncode import ReplaceImputeEncode, DT
from AdvancedAnalytics.Regression          import logreg, stepwise

# classes for Logistic regression
from sklearn.linear_model    import LogisticRegression
from sklearn.metrics         import r2_score
from sklearn.model_selection import train_test_split, cross_validate

# classes for Decision Tree Classification
from AdvancedAnalytics.Tree   import tree_classifier
from sklearn.tree             import DecisionTreeClassifier

# classes for Random Forest Classification
from AdvancedAnalytics.Forest import forest_classifier
from sklearn.ensemble         import RandomForestClassifier

# classes for sklearn FNN
from AdvancedAnalytics.NeuralNetwork import nn_classifier
from sklearn.neural_network          import MLPClassifier

with warnings.catch_warnings():  
    warnings.filterwarnings("ignore",category=FutureWarning)
    import tensorflow as tf
    from AdvancedAnalytics.NeuralNetwork import nn_classifier, nn_keras
    from keras.datasets                  import mnist
    from keras.utils                     import np_utils
    from tensorflow.keras.optimizers     import Adam
    from tensorflow.keras                import Sequential
    from tensorflow.keras                import regularizers
    from tensorflow.keras.layers         import Dense
    from tensorflow.keras                import models
    from tensorflow.keras                import layers
    from tensorflow.keras                import utils
    from tensorflow.keras.optimizers     import RMSprop, Adadelta, SGD
    
import matplotlib.pyplot as plt 

from math                    import log, isfinite, sqrt, pi
#*****************************************************************************
print("{:*>71s}".format('*'))

attribute_map = {
    'make':       [DT.Interval, (0.0, 100.0)],
    'address':    [DT.Interval, (0.0, 100.0)],
    'all':        [DT.Interval, (0.0, 100.0)],
    'W_3d':       [DT.Interval, (0.0, 100.0)],
    'our':        [DT.Interval, (0.0, 100.0)],
    'over':       [DT.Interval, (0.0, 100.0)],
    'remove':     [DT.Interval, (0.0, 100.0)],
    'internet':   [DT.Interval, (0.0, 100.0)],
    'order':      [DT.Interval, (0.0, 100.0)],
    'mail':       [DT.Interval, (0.0, 100.0)],
    'receive':    [DT.Interval, (0.0, 100.0)],
    'will ':      [DT.Interval, (0.0, 100.0)],
    'people':     [DT.Interval, (0.0, 100.0)],
    'report':     [DT.Interval, (0.0, 100.0)],
    'addresses':  [DT.Interval, (0.0, 100.0)],
    'free':       [DT.Interval, (0.0, 100.0)],
    'business':   [DT.Interval, (0.0, 100.0)],
    'email':      [DT.Interval, (0.0, 100.0)],
    'you ':       [DT.Interval, (0.0, 100.0)],
    'credit':     [DT.Interval, (0.0, 100.0)],
    'your':       [DT.Interval, (0.0, 100.0)],
    'font':       [DT.Interval, (0.0, 100.0)],
    'W_000':      [DT.Interval, (0.0, 100.0)],
    'money':      [DT.Interval, (0.0, 100.0)],
    'hp':         [DT.Interval, (0.0, 100.0)],
    'hpl':        [DT.Interval, (0.0, 100.0)],
    'george':     [DT.Interval, (0.0, 100.0)],
    'W_650':      [DT.Interval, (0.0, 100.0)],
    'lab':        [DT.Interval, (0.0, 100.0)],
    'labs':       [DT.Interval, (0.0, 100.0)],
    'telnet':     [DT.Interval, (0.0, 100.0)],
    'W_857':      [DT.Interval, (0.0, 100.0)],
    'data':       [DT.Interval, (0.0, 100.0)],
    'W_415':      [DT.Interval, (0.0, 100.0)],
    'W_85':       [DT.Interval, (0.0, 100.0)],
    'technology': [DT.Interval, (0.0, 100.0)],
    'W_1999':     [DT.Interval, (0.0, 100.0)],
    'parts':      [DT.Interval, (0.0, 100.0)],
    'pm':         [DT.Interval, (0.0, 100.0)],
    'direct':     [DT.Interval, (0.0, 100.0)],
    'cs':         [DT.Interval, (0.0, 100.0)],
    'meeting':    [DT.Interval, (0.0, 100.0)],
    'original':   [DT.Interval, (0.0, 100.0)],
    'project ':   [DT.Interval, (0.0, 100.0)],
    're:':        [DT.Interval, (0.0, 100.0)],
    'edu':        [DT.Interval, (0.0, 100.0)],
    'table ':     [DT.Interval, (0.0, 100.0)],
    'conference': [DT.Interval, (0.0, 100.0)],
    'C;':         [DT.Interval, (0.0, 100.0)],
    'C(':         [DT.Interval, (0.0, 100.0)],
    'C[':         [DT.Interval, (0.0, 100.0)],
    'C!':         [DT.Interval, (0.0, 100.0)],
    'C$':         [DT.Interval, (0.0, 100.0)],
    'C#':         [DT.Interval, (0.0, 100.0)],
    'CAP_avg':    [DT.Interval, (0.0, np.inf)],
    'CAP_long':   [DT.Interval, (0.0, np.inf)],
    'CAP_tot':    [DT.Interval, (0.0, np.inf)],
    'Spam':       [DT.Binary,   (0, 1)]
}

target = "Spam"
df = pd.read_csv("spambase.csv")

# RIE Code for each respective model type
RIE_Reg     = False
RIE_NonReg  = True

# Linear Regression Portion of Exam
SW          = False
Log_L1_CV   = False
Log_L1_Eval = False
Log_L2_CV   = False
Log_L2_Eval = False
Elastic_CV  = False
Elastic_Eval= False

# Decision Trees
Tree_CV     = False
Tree_Eval   = False
# Random Forest
Forest_CV   = False
Forest_Eval = False
# Neural Networks with sklearn and keras
FNN_CV      = False
FNN_Eval    = False
# 
Keras_Info  = True
Keras_CV    = True
Keras_Eval  = True


if RIE_Reg:
    print('\n******RIE for Logistic Regression Models*******')
    # There isn't any need to create a separate RIE for regularized or non
    # regularized regression as all attributes are interval, only target is 
    # binary plus it's already encoded
    rie        = ReplaceImputeEncode(data_map=attribute_map, display=True) 
    encoded_df = rie.fit_transform(df) # No missing values or outliers
    
    # These data have no mominal or binary attributes that require imputation or
    # one-hot encoding.  The target is Binary, but it is already encoded.
    y = df[target] 
    X = df.drop([target],axis=1)

if SW:
    sw = stepwise(df, target, reg="logistic", method="backward",
                        crit_in=0.1, crit_out=0.1, verbose=True)
    selected = sw.fit_transform()
    print("Number of Selected Features: ", len(selected))
    lr    = LogisticRegression(penalty='none', solver='lbfgs',
                              tol=1e-12, max_iter=10000)
    lr    = lr.fit(X[selected], y)
    proba = lr.predict_proba(X[selected])
    ll = 0
    misc = 0
    for i in range(y.shape[0]):
        if y[i] == 1:
            if proba[i,1] < 0.5:
                misc += 1
            d = log(proba[i,1])
        else:
            if proba[i,0] < 0.5:
                misc += 1
            d = log(proba[i,0])
        ll += d
    twoLL = -2.0*ll
    
    n     = y.shape[0]
    p     = len(selected)
    k     = len(lr.coef_)*(len(lr.coef_[0])+1) + 1
    bic   = twoLL + log(n)*k
    aic   = twoLL + 2*k
    R2 = r2_score(y, proba[:,1])
    adjr2 = 1.0-R2 
    adjr2 = ((n-1)/(n-p-1))*adjr2
    adjr2 = 1.0 - adjr2
    print("MISC: ", misc)
    print("Adj. R-Squared: ", adjr2)
    print("AIC: ", aic)
    print("BIC: ", bic)
    logreg.display_metrics(lr, X[selected], y)

if Log_L1_CV:
    print("******* Regularized Regression using L1 ************************")
    print("\n** Cross-Validation for Regularized (L1) Logistic Regression  **")
    print("** Hyperparameter  Optimization  based  on  Maximum  F1-Score **")
    score_list = ['accuracy', 'precision', 'recall',  'f1']
    C_list     = [1e-5, 9e-5, 1e-4, 2e-4, 0.5, 0.75, 1.0]
    best_f1    = 0
    best_c     = 1.0
    for c in C_list:
        lr     = LogisticRegression(C=c, penalty='l1', solver='saga', 
                                    max_iter=10000, tol=1e-4)
        scores = cross_validate(lr, X, y, scoring=score_list, cv=4,
                                return_train_score=False, n_jobs=-1)
    
        print("\n4-Fold CV for Logistic Regression with C=", c)
        print("{:.<18s}{:>6s}{:>13s}".format("Metric", "Mean", "Std. Dev."))
        for s in score_list:
            var = "test_"+s
            mean = scores[var].mean()
            std  = scores[var].std()
            print("{:.<18s}{:>7.4f}{:>10.4f}".format(s, mean, std))
            if s=='f1' and mean>best_f1:
                best_f1   = mean
                best_c    = c
    
if Log_L1_Eval:
    print("********* Best Logistic Regression (L1) *********")
    print("\nLogistic Regression with Best C= ", best_c)
    lr = LogisticRegression(C=best_c, penalty='l1', solver='saga',
                            tol=1e-4, max_iter=10000)
    lr = lr.fit(X, y)
    logreg.display_coef(lr, X, y, X.columns)
    logreg.display_metrics(lr, X, y)
    print('\n******BIC and AIC********')
    proba = lr.predict_proba(X)
    ll    = 0
    misc  = 0
    for i in range(y.shape[0]):
        if y[i] == 1:
            if proba[i,1] < 0.5:
                misc += 1
            d = log(proba[i,1])
        else:
            if proba[i,0] < 0.5:
                misc += 1
            d = log(proba[i,0])
        ll += d
    twoLL = -2.0*ll
    n = y.shape[0]
    p = X.shape[1]
    k = len(lr.coef_)*(len(lr.coef_[0])+1) + 1
    bic   = twoLL + log(n)*k
    aic   = twoLL + 2*k
    R2 = r2_score(y, proba[:,1])
    adjr2 = 1.0-R2 
    adjr2 = ((n-1)/(n-p-1))*adjr2
    adjr2 = 1.0 - adjr2
    print("MISC: ", misc)
    print("Adj. R-Squared: ", adjr2)
    print("AIC: ", aic)
    print("BIC: ", bic)
  
if Log_L2_CV:
    print("******* Regularized Regression using L2 ************************")
    print("\n** Cross-Validation for Regularized (L2) Logistic Regression  **")
    print("** Hyperparameter  Optimization  based  on  Maximum  F1-Score **")
    score_list = ['accuracy', 'precision', 'recall',  'f1']
    C_list     = [0.5, 0.6, 0.75, 1.0, 1.5]
    best_f1    = 0
    best_c     = 1.0
    for c in C_list:
        lr     = LogisticRegression(C=c, penalty='l2', solver='lbfgs', 
                                    max_iter=10000, tol=1e-4)
        scores = cross_validate(lr, X, y, scoring=score_list, cv=4,
                                return_train_score=False, n_jobs=-1)
    
        print("\n4-Fold CV for Logistic Regression with C=", c)
        print("{:.<18s}{:>6s}{:>13s}".format("Metric", "Mean", "Std. Dev."))
        for s in score_list:
            var = "test_"+s
            mean = scores[var].mean()
            std  = scores[var].std()
            print("{:.<18s}{:>7.4f}{:>10.4f}".format(s, mean, std))
            if s=='f1' and mean>best_f1:
                best_f1   = mean
                best_c    = c
    
if Log_L2_Eval:
    print("********* Best Logistic Regression (L2) *********")
    print("\nLogistic Regression with Best C= ", best_c)
    lr = LogisticRegression(C=best_c, penalty='l2', solver='lbfgs',
                            tol=1e-4, max_iter=10000)
    lr = lr.fit(X, y)
    logreg.display_coef(lr, X, y, X.columns)
    logreg.display_metrics(lr, X, y)
    print('\n******BIC and AIC********')
    proba = lr.predict_proba(X)
    ll    = 0
    misc  = 0
    for i in range(y.shape[0]):
        if y[i] == 1:
            if proba[i,1] < 0.5:
                misc += 1
            d = log(proba[i,1])
        else:
            if proba[i,0] < 0.5:
                misc += 1
            d = log(proba[i,0])
        ll += d
    twoLL = -2.0*ll
    n = y.shape[0]
    p = X.shape[1]
    k = len(lr.coef_)*(len(lr.coef_[0])+1) + 1
    bic   = twoLL + log(n)*k
    aic   = twoLL + 2*k
    R2 = r2_score(y, proba[:,1])
    adjr2 = 1.0-R2 
    adjr2 = ((n-1)/(n-p-1))*adjr2
    adjr2 = 1.0 - adjr2
    print("MISC: ", misc)
    print("Adj. R-Squared: ", adjr2)
    print("AIC: ", aic)
    print("BIC: ", bic)

if Elastic_CV:
    print("******* Regularized Regression using Elastic Net************************")
    print("\n** Cross-Validation for Elastic Net Regression  **")
    print("** Hyperparameter  Optimization  based  on  Maximum  F1-Score **")
    score_list = ['accuracy', 'precision', 'recall',  'f1']
    C_list     = [0.5, 0.75, 1.0]
    best_f1    = 0
    best_c     = 1.0
    l1_list    = [2e-4, 0.25, 0.5]
    for c in C_list:
        lr     = LogisticRegression(C=c, penalty='elasticnet', solver='saga', 
                                    max_iter=10000, tol=1e-4)
        scores = cross_validate(lr, X, y, scoring=score_list, cv=4,
                                return_train_score=False, n_jobs=-1)
        
        for l1 in l1_list:
            lr     = LogisticRegression(C=c, penalty='elasticnet', 
                                        solver='saga', l1_ratio=l1,
                                    max_iter=10000, tol=1e-4)
            scores = cross_validate(lr, X, y, scoring=score_list, cv=4,
                                return_train_score=False, n_jobs=-1)
            print("\n4-Fold CV for Logistic Regression with C=", c, 
                  'l1_ratio=', l1)
            print("{:.<18s}{:>6s}{:>13s}".format("Metric", "Mean", "Std. Dev."))
            for s in score_list:
                var = "test_"+s
                mean = scores[var].mean()
                std  = scores[var].std()
                print("{:.<18s}{:>7.4f}{:>10.4f}".format(s, mean, std))
                if s=='f1' and mean>best_f1:
                    best_f1   = mean
                    best_c    = c
                    best_l1 = l1
                    
if Elastic_Eval:
    lr = LogisticRegression(C=best_c, penalty='elasticnet', solver='saga', 
                            l1_ratio=best_l1,
                        max_iter=10000, random_state=12345)
    print("\nElastic Net Regression with Best C= ", best_c, " best l1_ratio= ",
          best_l1)
    lr = lr.fit(X, y)
    logreg.display_coef(lr, X, y, X.columns)
    logreg.display_metrics(lr, X, y)
    print('\n******BIC and AIC********')
    proba = lr.predict_proba(X)
    ll    = 0
    misc  = 0
    for i in range(y.shape[0]):
        if y[i] == 1:
            if proba[i,1] < 0.5:
                misc += 1
            d = log(proba[i,1])
        else:
            if proba[i,0] < 0.5:
                misc += 1
            d = log(proba[i,0])
        ll += d
    twoLL = -2.0*ll
    n = y.shape[0]
    p = X.shape[1]
    k = len(lr.coef_)*(len(lr.coef_[0])+1) + 1
    bic   = twoLL + log(n)*k
    aic   = twoLL + 2*k
    R2 = r2_score(y, proba[:,1])
    adjr2 = 1.0-R2 
    adjr2 = ((n-1)/(n-p-1))*adjr2
    adjr2 = 1.0 - adjr2
    print("MISC: ", misc)
    print("Adj. R-Squared: ", adjr2)
    print("AIC: ", aic)
    print("BIC: ", bic)
    
if RIE_NonReg:
# Encode for Non-Regression Models, not necessary to drop the last column as 
# all data is interval. 
    rie = ReplaceImputeEncode(data_map=attribute_map, display=True)
    encoded_df = rie.fit_transform(df)
    y    = encoded_df[target] # The target is not scaled or imputed
    y    = np.ravel(y)
    X    = encoded_df.drop(target,axis=1)
    cols = X.columns #required because numpy objects have no column labels
    X = np.array(X)
    opt_adad  = Adadelta(learning_rate=0.3, rho=0.95, epsilon = 1e-07)
    opt_adam  = Adam    (learning_rate=0.001,beta_1=0.9, beta_2=0.999, 
                                 epsilon=1e-7) 
    opt_sgd   = SGD(learning_rate=0.01, momentum=0.01, nesterov=True)
    opt_rms   = RMSprop(learning_rate=0.001, rho=0.9, momentum=0.01, 
                            centered=False, epsilon=1e-7)

if Tree_CV:
    print("***************** Decision Tree ************************")
    print("** 4-Fold Cross-Validation for Decision Tree Maximum Depth  **")
    print("** Hyperparameter Optimization  based  on  Maximum  F1-Score **")

    score_list = ['accuracy', 'precision', 'recall',  'f1']
    # 10-Fold Cross-Validation
    depths           = [18, 19, 20, 21, 22, 23, 24, 25]
    min_samples_list = [5, 7, 8, 9, 10, 11, 15]
    best             =  -1.0
    for d in depths:
        for leaf_size in min_samples_list:
            min_split_size = 2 * leaf_size
            print("\nTree Depth: ", d)
            dtc = DecisionTreeClassifier(max_depth = d, 
                                         min_samples_leaf  = leaf_size, 
                                         min_samples_split = min_split_size,
                                         random_state=12345)
            dtc = dtc.fit(X, y)
            scores = cross_validate(dtc, X, y, scoring=score_list, cv=4,
                                    return_train_score=False, n_jobs=-1)
            
            print("\nDecision Tree 4-Fold CV with Maximum Depth=", d,
                  " Min_Leaf_Size=", leaf_size, " Min_Split_Size=",
                  min_split_size)
            print("{:.<18s}{:>6s}{:>13s}"\
                  .format("Metric", "Mean", "Std. Dev."))
            for s in score_list:
                var = "test_"+s
                mean = scores[var].mean()
                std  = scores[var].std()
                print("{:.<18s}{:>7.4f}{:>10.4f}".format(s, mean, std))
                if s=='f1' and mean>best:
                    best       = mean
                    best_depth = d
                    best_leaf_size  = leaf_size
    
if Tree_Eval:
    print("************ Best Decision Tree ************")
    print("\nDecision Tree with Best Depth=", best_depth,
          " Best Min_Leaf_Size=", best_leaf_size)
    dtc = DecisionTreeClassifier(max_depth = best_depth, 
                                 min_samples_leaf  = best_leaf_size, 
                                 min_samples_split = 2*best_leaf_size,
                                 random_state=12345)
    dtc.fit(X, y)
    tree_classifier.display_importance(dtc, cols, top=15, plot=True)
    tree_classifier.display_metrics(dtc, X, y)

if Forest_CV:
    print("\n******** RANDOM FOREST ********")
    print('\n***********Choosing the Random Forest Parameters***************')
    # Cross-Validation
    score_list = ['accuracy', 'precision', 'recall',  'f1']
    estimators_list  = [50]
    min_samples_list = [5, 7, 8, 9, 10, 11, 15]
    depth_list       = [18, 19, 20, 21, 22, 23, 24, 25, None]
    features_list    = [5, 10, 20, 'auto', 32, None]
    best_score       = -1.0
    for e in estimators_list:
        for d in depth_list:
            for features in features_list:
                for leaf_size in min_samples_list:
                    split_size = 2*leaf_size
                    print("\nNumber of Trees: ", e, 
                          "Max_Depth: ", d,
                          "Max Features: ", features, 
                          "Min_Leaf_Size:", leaf_size)
                    rfc = RandomForestClassifier(n_estimators=e, 
                                criterion="gini",
                                min_samples_split=split_size, max_depth=d,
                                min_samples_leaf=leaf_size, 
                                max_features=features, 
                                n_jobs=-1, bootstrap=True, random_state=12345)
                    scores = cross_validate(rfc, X, y, scoring=score_list, \
                                            return_train_score=False, cv=4)
                    
                    print("{:.<20s}{:>6s}{:>13s}".format("Metric","Mean", 
                                                         "Std. Dev."))
                    for s in score_list:
                        var = "test_"+s
                        mean = scores[var].mean()
                        std  = scores[var].std()
                        print("{:.<20s}{:>7.4f}{:>10.4f}".format(s, mean, std))
                        if mean > best_score and s=='f1':
                            best_score      = mean
                            best_estimator  = e
                            best_depth      = d
                            best_features   = features
                            best_leaf_size  = leaf_size
                            best_split_size = split_size

if Forest_Eval:
    print("\nRandom Forest Based on Hyper Parameterization")
    print("Best Number of Trees (estimators) = ", best_estimator)
    print("Best Depth = ", best_depth)
    print("Best Leaf Size = ", best_leaf_size)
    print("Best Split Size = ", best_split_size)
    print("Best Max Features = ", best_features)
    rfc = RandomForestClassifier(n_estimators=best_estimator, 
                            criterion="gini", 
                            min_samples_split=best_split_size, 
                            max_depth=best_depth,
                            min_samples_leaf=best_leaf_size, 
                            max_features=best_features, 
                            n_jobs=-1, bootstrap=True, random_state=12345)
    rfc = rfc.fit(X, y)
    
    forest_classifier.display_metrics(rfc, X, y)
    forest_classifier.display_importance(rfc, cols, top=15, plot=True)

if FNN_CV:
    top_features = ['C$','remove','C!','free','CAP_avg','your','hp','CAP_long',
                    'CAP_tot','our','money','hpl','george','you','W_000']
    #Xs = X[top_features]
    print("******* Neural Network 4-Fold Cross-Validation ********")
    score_list = ['accuracy', 'precision', 'recall',  'f1']
    best = -np.inf
    # 4-Fold Cross-Validation
    neuron_list = [(3), (6), (7), (10), (20), (30), (3,3), (6,6), (7,7),
                   (10,5), (20,10), (30,15)]
    for neurons in neuron_list:
        print("\n FNN with : ", neurons, " perceptrons.")
        fnn = MLPClassifier(hidden_layer_sizes=neurons, solver='adam',
                        max_iter=100000, tol=1e-4, alpha=0.0,
                        random_state=12345)
        fnn.fit(X, y)
        scores = cross_validate(fnn, X, y, scoring=score_list, 
                                return_train_score=False, cv=4)
        
        print("{:.<18s}{:>6s}{:>13s}".format("Metric", "Mean", "Std. Dev."))
        for s in score_list:
            var = "test_"+s
            mean = scores[var].mean()
            std  = scores[var].std()
            print("{:.<18s}{:>7.4f}{:>10.4f}".format(s, mean, std))
            if s=='f1' and mean>best:
                best = mean
                best_neurons = neurons

if FNN_Eval:
    print("\nBest FNN Configuration: ", best_neurons)   
    Best_FNN = MLPClassifier(hidden_layer_sizes=best_neurons, solver='adam',
                        max_iter=100000, tol=1e-4,
                        alpha=0.0, random_state=12345)
    Best_FNN.fit(X, y) 
    nn_classifier.display_metrics(Best_FNN, X, y)
    
if Keras_Info:
    def keras_accuracy_plots(history_dict, predictt, predictv, 
                         yt, yv, y_out, file=None):
    
        epochs = len(history_dict['loss'])
        epoch_list   = range(1, epochs+1)
        plt.subplots(figsize=(12,10))
        plt.subplot(211)
        plt.plot(epoch_list, history_dict['loss'], 
                 'ro', label='Training Loss')
        plt.plot(epoch_list, history_dict["val_loss"], 
                 'b',  label='Validation Loss')
        plt.title("Loss vs. Percent Error")
        plt.ylabel("Loss")
        plt.legend(fontsize='xx-large')
        if file != None:
            f = "Loss" + file
            plt.savefig(f, format='pdf', dpi=180, 
                    bbox_inches='tight')
        plt.show()
        
        plt.subplots(figsize=(12,10))
        plt.subplot(212)
        plt.plot(epoch_list, history_dict['accuracy'], 
                 'ro', label='Training Accuracy')
        plt.plot(epoch_list, history_dict['val_accuracy'], 
                 'b',  label='Validation Accuracy')
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend(fontsize='xx-large')
        if file != None:
            f = "accuracy"+file
        plt.savefig(f, format='pdf', dpi=180, 
                    bbox_inches='tight')
        plt.show()
    
        print("         LOSS        ")
        print("---------------------")
        print("Training   ", round(history_dict["loss"][epochs-1], 4))
        print("Validation ", round(history_dict["val_loss"][epochs-1], 4))
        print("---------------------")
        
        n         = len(predictt)
        m         = y_out+1
        print("\n******** Confusion Matrix **********")
        print("------------------------------------\n")
        print("********  Training Data   **********")
        print("------------------------------------")
        conf = np.zeros((m, m), dtype='int32')
        misc = 0
        for i in range(n):
            kp = predictt[i,].argmax()
            ky = yt[i,].argmax()
            if predictt[i]>0.5:
                kp = 1
            else:
                kp = 0
            if yt[i] > 0.5:
                ky = 1
            else:
                ky = 0
            conf[ky, kp] += 1
            if ky != kp:
                misc += 1
        miscp = 100*misc/n
        for i in range(m):
            print("{:>5d} {:>5d}".format(conf[i,0], conf[i,1]))
        print("------------------------------------")
        print("Training Misclassification: {}/{}={:>5.2f}%".\
                  format(misc, n, miscp))
            
    
        n         = len(predictv)
        m         = y_out+1
        print("\n------------------------------------")
        print("******** Validation Data  **********")
        print("------------------------------------")
        conf = np.zeros((m, m), dtype='int32')
        misc = 0
        for i in range(n):
            kp = predictv[i,].argmax()
            ky = yv[i,].argmax()
            if predictt[i]>0.5:
                kp = 1
            else:
                kp = 0
            if yt[i] > 0.5:
                ky = 1
            else:
                ky = 0
            conf[ky, kp] += 1
            if ky != kp:
                misc += 1
        miscp = 100*misc/n
        for i in range(m):
            print("{:>5d} {:>5d}".format(conf[i,0], conf[i,1]))
        print("------------------------------------")
        print("Validation Misclassification: {}/{}={:>5.2f}%".\
                  format(misc, n, miscp))

if Keras_CV:
    neuron_list = [(2), (3), (4), (5), (6), (10), (2,1), (3,2), (4,3), (4,2),
                       (5,3)]
    best = -1e65
    n_features = X.shape[1]
    n_train    = X.shape[0]
    y_out      = 1 # for Binary or Interval Targets
    epochs     = 100
    size       = 3 # 0.5% of N
    for neurons in neuron_list:
        print("\nFNN with : ", neurons, " hidden layer perceptrons.")
        if type(neurons)==int:
            levels = 1
        else:
            levels = len(neurons)
        model = models.Sequential()
        if levels == 1:
            model.add(layers.Dense(neurons, input_shape=(n_features,), 
                               activation='relu', name='Hidden_Layer_1'))
        if levels>1:
            model.add(layers.Dense(neurons[0], input_shape=(n_features,), 
                               activation='relu', name='Hidden_Layer_1'))
            model.add(layers.Dense(neurons[1], activation='relu',    
                                                  name='Hidden_Layer_2'))
        if levels>2:
            model.add(layers.Dense(neurons[2], activation='relu',    
                                                  name='Hidden_Layer_3'))
        model.add(layers.Dense( y_out, activation='sigmoid', name='Output_Layer'))
        model.summary() # Display Neural Network Description
        
        model.compile(loss='binary_crossentropy', optimizer=opt_adad, 
                      metrics=['accuracy'])
        
        print("Train DataFrame:    N=", n_train)
        tf.random.set_seed(12345)
        history = model.fit(X, y, 
                            epochs=epochs, batch_size=size, verbose=1)
        dic = history.history
        acc = dic['accuracy'][epochs-1]
        if acc>best:
            best              = acc
            best_neurons      = neurons
            best_history_dict = dic
            predictX          = model.predict(X)
    print("**********************************************")
    
if Keras_Eval:
    print("Best Network used ", best_neurons, "neurons")
    neurons = best_neurons
    model = models.Sequential()
    if type(neurons)==int:
            levels = 1
    else:
        levels = len(neurons)
        
    if levels == 1:
        model.add(layers.Dense(neurons, input_shape=(n_features,), 
                           activation='relu', name='Hidden_Layer_1'))
    if levels>1:
        model.add(layers.Dense(neurons[0], input_shape=(n_features,), 
                           activation='relu', name='Hidden_Layer_1'))
        model.add(layers.Dense(neurons[1], activation='relu',    
                                              name='Hidden_Layer_2'))
    if levels>2:
        model.add(layers.Dense(neurons[2], activation='relu',    
                                              name='Hidden_Layer_3'))
    model.add(layers.Dense( y_out, activation='sigmoid', name='Output_Layer'))
    model.summary() # Display Neural Network Description
    
    model.compile(loss='binary_crossentropy', optimizer=opt_adad, 
                  metrics=['accuracy'])
    
    epochs     = 100
    size       = 3 # 0.5% N
    n_train    = X.shape[0]
    print("DataFrame:    N=", n_train)
    
    tf.random.set_seed(12345)
    history = model.fit(X, y, 
                        epochs=epochs, batch_size=size, verbose=1)
    
    history_dict = history.history
    predictX     = model.predict(X)
    acc          = dic['accuracy'][epochs-1]
    misc         = 1 - acc
    print("\n******** MISC **********")
    print('MISC :', misc)
    


    