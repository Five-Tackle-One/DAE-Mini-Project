import matplotlib
import seaborn as sns
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.utils import resample
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


def CI_test(model):
    t_c = stats.t.isf(0.025, model.df_resid)
    for i in range(0,len(model.params)):
        upperCI = model.params[i] + t_c*model.bse[i]
        lowerCI = model.params[i] - t_c*model.bse[i]
        
        if (model.params[i] > lowerCI) and (model.params[i] < upperCI):
            print('Variable: {}'.format(model.params.index[i]))
            print('Accept Hypothesis for: {}'.format(model.params[i]))
            print('CI: [ {}'.format(lowerCI) ) 
            print('  ' + str(upperCI) + ' ]')
            print('-------------------------------------')
        else:
            print('Variable: {}'.format(model.params.index[i]))
            print('Reject Hypothesis for: {}'.format(model.params[i]))
            print('CI: [ {}'.format(lowerCI))
            print(' ' + str(upperCI) + ' ]')
            print('-------------------------------------')
            
            
def stepwise_selection(X, y, 
                       initial_list=[], 
                       threshold_in=0.001, 
                       threshold_out = 0.05, 
                       verbose=True):
    """ Perform a forward-backward feature selection 
    based on p-value from statsmodels.api.OLS
    Parameters:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features 
    """
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included

"""

Prepare dataset for regression by one-hot encoding the dataset
"""
def encode_dataset(regress):
    X = regress.iloc[:,:-1]
    y = regress.iloc[:,-1]
    encode_columns = list(filter(lambda column:X[column].dtype == 'int64',np.asarray(X.columns)))
    X_encode = pd.get_dummies(X,columns=encode_columns,drop_first=False)
    encoded_data = X_encode.join(y)
    return encoded_data



"""
Perform regression on every predictor and return predictor with highest t-value

Input: Training Data
"""

def get_highest_predictor(training_data,predictors,target='net_income'):
    t_values = {}
    for predictor in predictors:
        formula = '{} ~ {}'.format(target,predictor)
        model = smf.ols(formula = formula,data=training_data).fit(intercept=False)
        if math.isnan(model.tvalues[predictor]) == False:
            t_values[predictor] = np.abs(model.tvalues[predictor])
    max_tvalue_predictor = sorted(t_values, key=t_values.get, reverse=True)[0]
    return max_tvalue_predictor


def stepwise_selection(X, y, 
                       initial_list=[], 
                       threshold_in=0.001, 
                       threshold_out = 0.05, 
                       verbose=True):
    """ Perform a forward-backward feature selection 
    based on p-value from statsmodels.api.OLS
    Parameters:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features 
    """
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed=True
            if verbose:
                pass
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                pass
        if not changed:
            break
    return included



def backward_selction(X,y):
    cols= list(X.columns)
    pmax = 1
    verbose=True
    while (len(cols)>0):
        p=[]
        x_1=pd.DataFrame(X[cols])
        x_1= sm.add_constant(x_1)
        model= sm.OLS(y,x_1).fit()
        p= pd.Series(model.pvalues.values[1:],index=cols)
        pmax= max(p)
        feature_with_p_max=p.idxmax()
        if(pmax>0.05):
            cols.remove(feature_with_p_max)
        else:
            break
    selected_features= cols
    selected_X = X[selected_features]
    selected_y = y
    selected_data = selected_X.join(selected_y)
    return selected_data



def calculate_vif_(X, thresh):
    """Calculation of VIFs to reduce multicollinearity problems.

    Parameters:
    -----------
    X : pandas DataFrame with all possible predictors but
    not the response variable.

    thresh: integer, threshold by which to exclude variables
    for dataset

    Returns:
    --------
    remaining: list of variable names that still remain in 
               the dataset

    """
    cols = X.columns
    variables = np.arange(X.shape[1])
    dropped=True
    while dropped:
        dropped=False
        c = X[cols[variables]].values
        vif = [variance_inflation_factor(c, ix) for ix in np.arange(c.shape[1])]
        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            variables = np.delete(variables, maxloc)
            dropped=True
    remaining=cols[variables]
    return remaining


def remove_inflated_variables(train_X, sample_size, thresh = 10):
    """Calculation of VIFs to reduce multicollinearity problems
    using a subset of the data

    Parameters:
    -----------
    train_data : pandas DataFrame with all possible predictors and
    without the response variable

    sample_size: integer, size of the sample used to estimate the
    VIFs

    Returns:
    --------
    remaining: list of variable names that still remain in 
               the dataset

    """
    remaining = calculate_vif_(train_X, thresh)
    return remaining

def forward_selected(data, response):
    """Linear model designed by forward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = 0.0, 0.0
    count=0
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ + {} + 1".format(response,
                                           ' + '.join(selected + [candidate]))
            score = smf.ols(formula, data).fit(fit_intercept=False).rsquared_adj
            scores_with_candidates.append((score, candidate))
        count+=1
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if ((current_score < best_new_score) or (count>=20)):
            count=0
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    formula = "{} ~ {} + 1".format(response,' + '.join(selected))
    model = smf.ols(formula, data).fit(fit_intercept=False)
    return model


def bootstrapping(train_data, simulations, sample_size):
    """Use random sampling to collect the features that are most often
    used for the optimal regression model 

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and
    the response variable
    
    simulations : integer, the number of samples that must be taken
    from the data

    sample_size: integer, size of the sample used to fit the
    regression model

    Returns:
    --------
    resulting_models: list of the models and model parameters fit 
    for each sample

    """
    temp_X = train_data.drop('net_income', axis = 1)
    temp_y = train_data['net_income']
    optimal_models = []
    for i in range(0,simulations):
        remaining = remove_inflated_variables(train_data, sample_size=sample_size, thresh=5)
        X = train_data[remaining]
        y = train_data['net_income']
        result_data = backward_selction(X, y) 
        model = forward_selected(result_data, 'net_income')
        choosen=[]
        choosen=[model,model.params,model.bse,model.pvalues,model.conf_int(),model.condition_number,model.rsquared_adj]
        optimal_models.append(choosen)

    return optimal_models



def most_common_features(models,adj_R_squared,iters_fraction):
    """Use model list obtained from resampling procedure and 
    select the features that were used in the best models

    Parameters:
    -----------
    models : list of the models and model parameters fit 
    for each sample

    adj_R_squared: float, the threshold value for the R-squared adjusted 
    value for a model to be included in feature list
    
    iters_fraction : integer, minimum number of models in which a 
    feature must be included for it to be included in the feature 
    list

    Returns:
    --------
    valid: list of the features that meet all the necessary
    criteria to be included in the final model

    """
    features = {}
    for i in range(len(models)):
        if (models[i][6] >= adj_R_squared):
            feat = list(models[i][1].index)
            for f in feat:
                try:
                    features[f]+=1
                except:
                    features[f]=1
    valid = []
    for feat,value in features.items():
#         print(value)
        if ((value >= iters_fraction) and (feat != 'Intercept')):
            valid.append(feat)
    return valid