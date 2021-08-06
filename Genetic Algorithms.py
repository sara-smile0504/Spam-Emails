#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@target: Binary, Spam already encoded
@author: Sara Foster
"""

from deap import creator, base, tools, algorithms

import random, sys, time, warnings
import pandas as pd
import numpy  as np
import matplotlib.pyplot               as plt
import statsmodels.api                 as sm
import statsmodels.tools.eval_measures as em

from AdvancedAnalytics.ReplaceImputeEncode import ReplaceImputeEncode, DT
from AdvancedAnalytics.Regression          import logreg, stepwise

from sklearn.linear_model    import LogisticRegression, LinearRegression
from sklearn.metrics         import mean_squared_error, r2_score
# classes for sklearn validation
from sklearn.model_selection import train_test_split, cross_validate
from scipy.linalg            import qr_multiply, solve_triangular
from math                    import log, isfinite, sqrt, pi
              
def rngFit(z):
    r = maxFit(z) - minFit(z)
    return round(r, 3)

def avgFit(z):
    tot = 0.0
    cnt = 0
    for i in range(len(z)):
        if isfinite(z[i][0]):
            tot += z[i][0]
            cnt += 1
    if cnt>0:
        return round(tot/cnt, 4)
    else:
        return np.nan

def maxFit(z):
    maximum = 0
    for i in range(len(z)):
        if z[i][0] > maximum:
            maximum = z[i][0]
    return round(maximum, 5)

def minFit(z):
    minimum = np.inf
    for i in range(len(z)):
        if z[i][0] < minimum:
            minimum = z[i][0]
    return round(minimum, 5)

def cvFit(z):
    avg = avgFit(z)
    std = stdFit(z)
    if isfinite(avg):
        return round(100*std/avg, 3)
    else:
        return np.nan

def logMinFit(z):
    try:
        return round(log(minFit(z)), 5)
    except:
        return -np.inf
    
def logMaxFit(z):
    try:
        return round(log(maxFit(z)), 5)
    except:
        return np.inf

def stdFit(z):
    sum1  = 0.0
    sum2  = 0.0
    cnt   = 0
    for i in range(len(z)):
        if isfinite(z[i][0]):
            sum1 += z[i][0]
            sum2 += z[i][0] * z[i][0]
            cnt += 1
    if cnt < 2:
        return np.nan
    else:
        sumsq = (sum1*sum1)/cnt
        return round(sqrt((sum2 - sumsq)/(cnt-1)), 4)
def features_min(z):
    minimum = np.inf
    feature = np.inf
    for i in range(len(z)):
        if z[i][0] < minimum:
            minimum = z[i][0]
            feature = z[i][1]
        if z[i][0] == minimum and z[i][1] < feature:
            feature = z[i][1]
    return round(feature,0)

def features_max(z):
    maximum = -np.inf
    feature =  np.inf
    for i in range(len(z)):
        if z[i][0] > maximum:
            maximum = z[i][0]
            feature = z[i][1]
        if z[i][0] == maximum and z[i][1] < feature:
            feature = z[i][1]
    return round(feature,0)

def geneticAlgorithm(X, y, n_population, n_generation, p_select=None, 
                     method=None, reg=None, goodFit=None,  calcModel=None,
                     n_int=None, n_nom=None, n_frac=None):
    """
    Deap global variables
    Initialize variables to use eaSimple
    """
    if method==None:
        method = 'random'
    if method=='features':
        if type(p_select) == float:
            if p_select>1.0 or p_select<0.0:
                raise ValueError("method='features' requires 0<p_select<1")
                sys.exit()
        else:
            raise ValueError("method='featires' requires 0<p_select<1")
            sys.exit()
                
    if goodFit==None:
        goodFit='bic'
    if calcModel==None:
        calcModel='statsmodels'
    if type(y)==np.ndarray:
        nval = len(np.unique(y))
    else:
        nval = y.nunique()
    if reg==None:
        if nval > 20:
            reg = 'linear'
        else: 
            reg = 'logistic'
    if goodFit.lower()!='adjr2':
        opt = -1.0 # Minimize goodness of fit
    else:
        opt =  1.0 # Maximize goodness of fit
 # create individual fitness dictionary
    ifit = {}
    # create individual
    # Two weights for two optimization (goodness of fit, number of features)
    # A negative weight indicates minimize that function.
    # A positive weight indicates maximize that function.
    with warnings.catch_warnings():  
        warnings.filterwarnings("ignore",category=RuntimeWarning)
        creator.create("FitnessMax", base.Fitness, weights=(opt, -1.0))
        creator.create("Individual", list, fitness=creator.FitnessMax)

    # create toolbox
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("population_guess", initPopulation, list, 
                                                      creator.Individual)
    toolbox.register("individual", tools.initRepeat, creator.Individual, 
                                     toolbox.attr_bool, n=len(X.columns))
    toolbox.register("population", tools.initRepeat, list,
                                                      toolbox.individual)
    if   reg.lower()=='logistic':
        toolbox.register("evaluate", evalFitnessLogistic, X=X, y=y, 
                         goodFit=goodFit, calcModel=calcModel, ifit=ifit)
    elif reg.lower()=='linear':
        toolbox.register("evaluate", evalFitnessLinear, X=X, y=y, 
                         goodFit=goodFit, calcModel=calcModel, ifit=ifit)
    else:
        raise ValueError("reg not set to 'linear' or 'logistic'")
        sys.exit()
    toolbox.register("mate",     tools.cxTwoPoint)
    toolbox.register("mutate",   tools.mutFlipBit, indpb=0.02)
    toolbox.register("select",   tools.selTournament, tournsize=7)

    if method=='random':
        pop = toolbox.population(n_population)
    else:
        # initialize parameters
        # n_int Total number of interval features
        # n_nom List of number of dummy variables for each categorical var
        pop = toolbox.population_guess(method, n_int, n_nom, n_frac, 
                                       n_population, p_select, X, y)
        #n_population = len(pop)
    hof   = tools.HallOfFame(n_population * n_generation)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    if goodFit.lower()!='adjr2':
        stats.register("features", features_min)
    else:
        stats.register("features", features_max)
    stats.register("range",    rngFit)
    stats.register("min",      minFit)
    stats.register("avg",      avgFit)
    stats.register("max",      maxFit)
    if goodFit.lower()!='adjr2':
        stats.register("Ln(Fit)",  logMinFit)
    else:
        stats.register("Ln(Fit)",  logMaxFit)
        

    # genetic algorithm
    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.9, mutpb=0.5,
                                   ngen=n_generation, stats=stats, 
                                   halloffame=hof, verbose=True)

    # return hall of fame
    return hof, logbook

def evalFitnessLinear(individual, X, y, goodFit, calcModel, ifit):
    # returns (goodness of fit, number of features)
    cols  = [index for index in range(len(individual)) 
            if individual[index] == 1 ]# get features subset, 
                                       # drop features with cols[i] != 1
    if type(X)==np.ndarray:
        X_selected = X[:, cols]
    else:
        X_selected = X.iloc[:, cols]
    
    features = X_selected.shape[1]
    n = X_selected.shape[0]
    p = features
    k = features + 2 # 2 for intercept and variance
    ind = ""     
    for i in range(len(individual)):
        if individual[i] == 0:
            ind += '0'
        else:
            ind += '1'
    try:
        fit = ifit[ind]
        return(fit, features)
    except:
        pass
    goodFit   = goodFit.lower()
    calcModel = calcModel.lower()
    if   k > n+2 and goodFit=='bic':
        return (np.inf, features)
    elif k > n+2 and goodFit=='adjr2':
        return (0, features)
    
    if calcModel == "qr_decomp":
        Xc     = sm.add_constant(X_selected)
        qty, r = qr_multiply(Xc, y)
        coef   = solve_triangular(r, qty)
        pred   = (Xc @ coef)
        resid  = pred - y
        ASE    = (resid @ resid) / n
        if ASE > 0:
            twoLL = n*(log(2*pi) + 1.0 + log(ASE))
            bic   = twoLL + log(n)*k
            aic   = twoLL + 2*k
            R2    = r2_score(y, pred)
            if R2 > 0.99999:
                bic = -np.inf
        else: 
            bic = -np.inf
            aic = -np.inf
            R2  = 1.0
            
        if goodFit == 'bic':
            return(bic, features)
        elif goodFit == 'aic':
            return(aic, features)
        elif goodFit == 'misc':
            raise ValueError("Fitness 'misc' only used with Binary targets")
            sys.exit()
        else:
            adjr2 = 1.0-R2 
            adjr2 = ((n-1)/(n-p-1))*adjr2
            if adjr2 < 1.0:
                adjr2 = 1.0 - adjr2
                return(adjr2, features)
            else:
                return(0.0, features)

    elif calcModel== "statsmodels":
        Xc       = sm.add_constant(X_selected)
        model    = sm.OLS(y, Xc)
        results  = model.fit()
        parms    = np.ravel(results.params)
        if goodFit == "adjr2":
            pred  = model.predict(parms)
            R2    = r2_score(y, pred)
            adjr2 = 1.0-R2 
            adjr2 = ((n-1)/(n-p-1))*adjr2
            if adjr2 < 1.0:
                adjr2 = 1.0 - adjr2
                return(adjr2, features)
            else:
                return(0.0, features)
        else:
            loglike  = model.loglike(results.params)
            model_df = model.df_model + 2 #plus intercept and sigma
            nobs     = y.shape[0]
            if goodFit=='bic':
                bic  = em.bic(loglike, nobs, model_df)
                return(bic, features)
            elif goodFit=='aic':
                aic  = em.aic(loglike, nobs, model_df)
                return(aic, features)
            else:
                raise ValueError("Fitness 'misc' only used with Binary targets")
                sys.exit()
        
    elif calcModel=='sklearn':
        # sklearn linear regression does not handle no features
        if X_selected.shape[1]>0:
            lr   = LinearRegression().fit(X_selected,y)
            pred = lr.predict(X_selected)
        else:
            avg  = y.mean()
            pred = np.array([avg]*y.shape[0])
        ASE  = mean_squared_error(y,pred)
        if ASE > 0:
            twoLL = n*(log(2*pi) + 1.0 + log(ASE))
            bic   = twoLL + log(n)*k
            aic   = twoLL + 2*k
            R2 = r2_score(y, pred)
            if R2 > 0.99999:
                bic = -np.inf
        else: 
            R2  = r2_score(y, pred)
            bic = -np.inf
            aic = -np.inf
            
        if goodFit == 'bic':
            return(bic, features)
        elif goodFit=='aic':
            return(aic, features)
        elif goodFit=='adjr2':
            adjr2 = 1.0-R2 
            adjr2 = ((n-1)/(n-p-1))*adjr2
            if adjr2 < 1.0:
                adjr2 = 1.0 - adjr2
                return(adjr2, features)
            else:
                return(0.0, features)
        else:
            raise ValueError("Goodness of Fit ",
                             goodFit, "not available for interval targets.")
            sys.exit()
    else:
        raise ValueError("calcModel not 'statsmodels', 'sklearn', or 'QR_decomp'")
        sys.exit()
    
def evalFitnessLogistic(individual, X, y, goodFit, calcModel, ifit):
    # Number of categories in y
    if type(y)==np.ndarray:
        n_cat = len(np.unique(y))
    else:
        n_cat = y.nunique()
    # returns (goodness of fit, number of features)
    cols  = [index for index in range(len(individual)) 
            if individual[index] == 1 ]# get features subset, 
                                       # drop features with cols[i] != 1
    if type(X)==np.ndarray:
        X_selected = X[:, cols]
    else:
        X_selected = X.iloc[:, cols]
    
    features = X_selected.shape[1]
    n = X_selected.shape[0]
    p = features
    if n_cat <= 2:
        k = features + 2 #for intercept and variance
    else:
        k = n_cat*(features + 1) + 1 # n_cat intercepts and +1 for variance
    ind = ""     
    for i in range(len(individual)):
        if individual[i] == 0:
            ind += '0'
        else:
            ind += '1'
    try:
        # See if this individual was previouly evaluated
        fit = ifit[ind]
        # If so, return the stored fitness
        return(fit, features)
    except:
        # If this individual is not in ifit, evaluate and store in ifit[ind]
        pass
    goodFit   = goodFit.lower()
    calcModel = calcModel.lower()
    # If number of features selected (k) greater than n, return worst fitness
    if   k > n+2 and goodFit=='bic':
        return (np.inf, features)
    elif k > n+2 and goodFit=='adjr2':
        return (0, features)
    
    if calcModel== "statsmodels":
        Xc = sm.add_constant(X_selected)
        try:
            model   = sm.Logit(y, Xc)
            results = model.fit(disp=False) 
        except:
            print("Singular Fit Encountered with", features, "features")
            if goodFit != 'adjr2':
                return(-np.inf, features)
            else:
                return(1.0, features)
        proba   = model.predict(results.params)   
        if goodFit == "adjr2":
            R2    = r2_score(y, proba)
            adjr2 = 1.0-R2 
            adjr2 = ((n-1)/(n-p-1))*adjr2 # p == number of features
            if adjr2 < 1.0:
                adjr2 = 1.0 - adjr2
                return(adjr2, features)
            else:
                return(0.0, features)
        else:
            ll = 0
            misc = 0
            for i in range(n):
                if y[i] == 1:
                    if proba[i] < 0.5:
                        misc += 1
                    d = log(proba[i])
                else:
                    if proba[i] > 0.5:
                        misc += 1
                    d = log(1.0 - proba[i])
                ll += d
            if goodFit=='bic':
                bic  = em.bic(ll, n, k)
                return(bic, features)
            elif goodFit=='aic':
                aic  = em.aic(ll, n, k)
                return(aic, features)
            elif goodFit=='misc':
                misc = 100.0*misc/n
                return(misc, features)
            else:
                raise ValueError(goodFit, 
                                 " is unsupported goodness of fit criterion")
                sys.exit()
        
    elif calcModel=='sklearn':
        # sklearn linear regression does not handle no features
        if X_selected.shape[1]>0:
            if X_selected.shape[0]*X_selected.shape[1] > 50000:
                opt='lbfgs'
            else:
                opt = 'newton-cg'
            lr    = LogisticRegression(penalty='none', solver=opt,
                                  tol=1e-4, max_iter=50000)
            lr    = lr.fit(X_selected, y)
            proba = lr.predict_proba(X_selected)
        else:
            # Return numpy array (n x 2) with all 0.5
            proba = np.full((y.shape[0],2),0.5)
        ll = 0
        misc = 0
        for i in range(n):
            if y[i] == 1:
                if proba[i,1] < 0.5:
                    misc += 1
                d = log(proba[i,1])
            else:
                if proba[i,0] < 0.5:
                    misc += 1
                if proba[i,0] > 0:
                    d = log(proba[i,0])
                else:
                    d = log(1e-16)
            ll += d
        twoLL = -2.0*ll
        if goodFit == 'bic':
            bic   = twoLL + log(n)*k
            return(bic, features)
        elif goodFit=='aic':
            aic   = twoLL + 2*k
            return(aic, features)
        elif goodFit=='misc':
            misc_rate = misc
            return(misc_rate, features)
        elif goodFit=='adjr2':
            R2    = r2_score(y, proba[:,1])
            adjr2 = 1.0-R2 
            adjr2 = ((n-1)/(n-p-1))*adjr2
            if adjr2 < 1.0:
                adjr2 = 1.0 - adjr2
                return(adjr2, features)
            else:
                # AdjR2 is Negative
                return(0.0, features)
        else:
            raise ValueError(goodFit, 
                             " unsupported goodness of fit criterion")
            sys.exit()
    else:
        raise ValueError(calcModel, " unsupported calculation model")
        sys.exit()
    
def initPopulation(pcls, ind_init, method, 
                   n_int, n_nom, n_frac, n_pop, p_select, X, y):
    k = X.shape[1] # total number of features in X
    
    if method == 'features':
        print(" ")
        print("{:*>71s}".format('*'))
        print("{:*>14s}   PROPORTION OF FEATURES  {:*>30s}". format('*', '*'))
        print("{:*>71s}".format('*'))
        # Include all features except p_drop%, the bottom percent
        p_drop = 1.0 - p_select
        for i in range(n_pop):
            icls = [1]*k
            for j in range(k):
                rnd = np.random.uniform()
                if rnd < p_drop:
                    icls[j] = 0
            ind = ind_init(icls)
            if i > 0:
                pcls.append(ind)
            else:
                pcls = [ind]
            
    elif method == 'backward':
        print(" ")
        print("{:*>71s}".format('*'))
        print("{:*>14s}     STEPWISE SELECTION    {:*>30s}". format('*', '*'))
        print("{:*>71s}".format('*'))
        sw         = stepwise(df, target, reg="logistic", method="backward",
                        crit_in=0.1, crit_out=0.1, verbose=True)
        selected   = sw.fit_transform()
        n_selected = len(selected)
        print("Number of Selected Features: ", n_selected)
        icls = [0]*k
        for i in range(k):
            for j in range(n_selected):
                if X.columns[i] == selected[j]:
                    icls[i] = 1
                    break
        ind  = ind_init(icls)
        pcls = [ind]
        icls_stepwise = icls[:]
        m = k - n_selected + 10
        for j in range(m):
            icls = icls_stepwise[:]
            for i in range(k):
                if icls[i] == 0:
                    rnd = np.random.uniform()
                    if rnd < 0.3:
                        icls[i] = 1
            ind     = ind_init(icls)
            pcls.append(ind)
                
    elif method == 'star' :
        # Initialize Null Case (no features)
        icls = [0]*k
        ind  = ind_init(icls)
        pcls = [ind]
        # Add "All" one-feature selection (star points)
        for i in range(k):
            icls = [0]*k
            icls[i]  = 1
            ind = ind_init(icls)
            pcls.append(ind)
            
    else:
        raise ValueError(method, " unsupported initialization method")
        sys.exit()
    
    return pcls

def findBest(hof, goodFit, X, y, top=1):
    #Find Best Individual in Hall of Fame
    print("Individuals in HoF: ", len(hof))
    #if top==None:
    #    top=1
    goodFit  = goodFit.lower()
    features = np.inf
    if goodFit=='bic' or goodFit=='aic' or goodFit=='misc':
        bestFit = np.inf
        for individual in hof:
            if(individual.fitness.values[0] < bestFit):
                bestFit     = individual.fitness.values[0]
                _individual = individual
            if (sum(individual) < features and 
                individual.fitness.values[0] == bestFit):
                features    = sum(individual)
                _individual = individual
    elif goodFit=='adjr2':
        bestFit = -np.inf
        for individual in hof:
            if(individual.fitness.values[0] > bestFit):
                bestFit     = individual.fitness.values[0]
                _individual = individual
            if (sum(individual) < features and 
                (individual.fitness.values[0] == bestFit or
                   individual.fitness.values[0] > 0.9999)):
                features    = sum(individual)
                _individual = individual
    else:
        raise RuntimeError(goodFit, " unsupported goodnes of fit criterion")
        sys.exit()
    if type(X)==np.ndarray:
        z = np.ravel(_individual)
        z = z.nonzero()
        _individualHeader = z[0]
    else:
        _individualHeader = [list(X)[i] for i in range(len(_individual)) 
                        if _individual[i] == 1]
    return _individual.fitness.values, _individual, _individualHeader

        
def plotGenerations(gen, lnbic, features):
    fig, ax1 = plt.subplots()
    ax1.set_xlabel("GA GENERATION", fontsize="x-large",fontweight="heavy")
    ax1.tick_params(axis='x', labelcolor="black", labelsize="x-large")
    ax1.tick_params(axis='y', labelcolor="green", labelsize="x-large")
    #ax1.set_xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    ax1.set_ylabel("Log(Fit)", fontsize="x-large", fontweight="heavy", 
                   color="green")
    ax1.set_facecolor((0.95,0.95,0.95))
    #ax1.grid(axis='x', linestyle='--', linewidth=1, color='gray')
    ax1.plot(gen, lnbic, 'go-', color="green", 
                         linewidth=2, markersize=10)
    ax2 = ax1.twinx()
    ax2.tick_params(axis='y', labelcolor="blue", labelsize="x-large")
    ax2.set_ylabel("Number of Features Selected", fontsize="x-large", 
                   fontweight="heavy", color="blue")
    #ax2.set_yticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30])
    #ax2.grid(axis='y', linestyle='--', linewidth=1, color='gray')
    ax2.plot(gen, features, 'bs-', color="blue", 
                         linewidth=2, markersize=10)
    plt.savefig("GA_Feature_Select.pdf")
    plt.show()
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
print("Read", df.shape[0], "observations with", 
      df.shape[1], "attributes:\n")
rie = ReplaceImputeEncode(data_map=attribute_map, display=True) 
encoded_df = rie.fit_transform(df) # In this case, no missing or outliers

y = encoded_df[target] # The target is not scaled or imputed
X = encoded_df.drop(target, axis=1)

print("{:*>71s}".format('*'))
# apply genetic algorithm
# n_init:  set to the number of candidate interval and binary features
# n_nom:   set to a list of levels for each candidate nominal feature
#          if there are no candidate nominal features, set to an empty list []
n_int = 60     # Interval and binary features
n_nom = 0      # Nominal Variables
if type(n_nom)==int:
    p = n_int
elif type(n_nom)==list:
    p = n_int + sum(n_nom)  # Total number of features 60
else:
    raise ValueError("n_nom is not a list")
    sys.exit()

# models:  available regression models
# fitness: available fitness criteria. MISC is only available for Binary
#          targets
# init:    available Generation Zero initialization algorithms.  
           # [0] 'star' individuals are set with only one feature enabled
           # [1] 'random' individuals are set with randomly selected features
           # [2] 'backward' individuals are set to using backward stepwise
           # [3] 'features' individuals are set with randomly selected
           #     features with selection probablility determined by p_features
models     = ['sklearn', 'statsmodels', 'QR_decomp']
fitness    = ['bic', 'aic', 'adjR2', 'misc']
init       = ['star', 'random', 'backward', 'features']

calcModel  = models [0] # selected regression model
goodFit    = fitness[3] # selected fitness criterion
initMethod = init[3]    # selected Gen Zero initialization method
             
# n_pop is Generation Zero size.  Subsequent generations will be near
#       this size.
# n_gen is the number of generations after Gen Zero, each progressively better 
#       than the previous.  This needs to be large enough to allow the 
#       search algorithm to select the optimum features.
# p_features is the approximate proportion of features randomly selected
#       for each individual in Generation Zero.  This is only allowed 
#       with the 'features' initialization algorithm.
#       
# Note: This algorithm optimizes the fitness of the individual while 
#       minimizing the number of selected features.

if initMethod=='star':
    n_pop = p+1 # number of features + 1
    n_gen = 35
elif initMethod=='features':
    n_pop = 30
    n_gen = 20
    p_features = 0.8 # proportion of features in Gen zero
elif initMethod == 'backward':
    n_pop = 30
    n_gen = 10
elif initMethod == 'random':
    n_pop = 30
    n_gen = 10
else:
    raise ValueError(initMethod, " unsupported initialization method")
    sys.exit()

if initMethod != 'features':
    p_features = None
print("{:*>71s}".format('*'))
print("{:*>14s}     GA Selection using {:>5s} Fitness         {:*>11s}". 
      format('*', goodFit, '*'))
print("{:*>14s} {:>11s} Models and {:>6s} Initialization {:*>11s}". 
      format('*', calcModel, initMethod, '*'))
print(" ")
random.seed(12345)
start = time.time()
hof, logbook = geneticAlgorithm(X, y, n_pop, n_gen, p_select=p_features, 
                                method=initMethod, reg='logistic', 
                                goodFit=goodFit, calcModel=calcModel, 
                                n_int=n_int, n_nom=n_nom)

gen, features, min_, avg_, max_, rng_, lnfit = logbook.select("gen",
                    "features", "min", "avg", "max", "range", "Ln(Fit)")
end = time.time()    
duration = end-start
print("GA Runtime ", duration, " sec.")

# Plot Fitness and Number of Features versus Generation
plotGenerations(gen, lnfit, features)

# select the best individual
fit, individual, header = findBest(hof, goodFit, X, y)
print("Best Fitness:", fit[0])
print("Number of Features Selected: ", len(header))
print("\nFeatures:", header)

# sklearn Logistic Regression is sensitive to solver selection
# This code reproduces the solver selection in the GA search

n    = y.shape[0]
p    = len(header)
size = n*p
if size > 50000:
    opt='lbfgs'
else:
    opt='newton-cg'
lr    = LogisticRegression(penalty='none', solver=opt,
                          tol=1e-4, max_iter=10000)
lr    = lr.fit(X[header], y)
proba = lr.predict_proba(X[header])
ll    = 0
misc  = 0
for i in range(n):
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
k     = p + 2
bic   = twoLL + log(n)*k
aic   = twoLL + 2*k
R2 = r2_score(y, proba[:, 1])
adjr2 = 1.0-R2 
adjr2 = ((n-1)/(n-p-1))*adjr2
adjr2 = 1.0 - adjr2
print("MISC: ", misc)
print("Adj. R-Squared: ", adjr2)
print("AIC: ", aic)
print("BIC: ", bic)
logreg.display_metrics(lr, X[header], y)

print(" ")
print("{:*>71s}".format('*'))
print("{:*>14s}     STEPWISE SELECTION    {:*>30s}". format('*', '*'))
print("{:*>71s}".format('*'))

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

print(" ")
print("{:*>71s}".format('*'))
print("{:*>14s}     FIT FULL MODEL        {:*>30s}". format('*', '*'))
print("{:*>71s}".format('*'))
lr   = LogisticRegression(penalty='none', solver='lbfgs',
                          tol=1e-12, max_iter=10000)
lr    = lr.fit(X, y)
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
logreg.display_metrics(lr, X, y)


