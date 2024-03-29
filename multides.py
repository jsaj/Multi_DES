from pandas.io.formats.style_render import DataFrame
import pandas as pd
import numpy as np
import re

from warnings import filterwarnings
from sklearn.model_selection import train_test_split

# base classifiers
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import GaussianNB

# import ds techniques
from deslib.des import KNORAU

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, accuracy_score
from scipy import *
from scipy.stats import rankdata
from imblearn.metrics import geometric_mean_score
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import confusion_matrix

class MULTIDES(object):
  """Class for Multi-stage Dynamic Ensemble Selection (Multi-DES) for Cross-Project Defect Prediction.

  References
  ----------

    R. M. Cruz, L. G. Hafemann, R. Sabourin, and G. D. Cavalcanti.
    "Deslib:  A dynamicensemble selection library in python."
    Journal of Machine Learning Research, vol. 21, no. 8, pp. 1–5, 2020.
  
  """
  def __init__(self, bug_dataset,
               dynamic_algorithm=None,
               base_estimator=None,
               size_pool=None,
               selection_approach=None):
    
    if type(dynamic_algorithm) != list and dynamic_algorithm == None:
      dynamic_algorithm = [KNORAU()]
    if type(dynamic_algorithm) != list:
      dynamic_algorithm = [dynamic_algorithm]
    
    if type(base_estimator) != list and base_estimator != None:
      base_estimator = [base_estimator]
    elif base_estimator == None:
      base_estimator = [GaussianNB()]
    elif type(base_estimator) != list:
      base_estimator = [base_estimator]

    if selection_approach == None:
      selection_approach = 'selection'
    elif selection_approach in ['f1', 'auc', 'pf']:
      selection_approach = selection_approach
    else:
      raise ImportError(
        'Selection approach not available.'
        'Please check options and try again.')
    
    if type(size_pool) != list and size_pool == None:
      size_pool = [10]
    elif type(size_pool) == list and type(size_pool[0]) == int:
      size_pool = size_pool


    base_classifiers_dict = dict()
    for clf in base_estimator:
      name_clf = str(clf).split('(')[0]
      base_classifiers_dict[name_clf] = clf

    dynamic_algorithms_dict = dict()
    for ds in dynamic_algorithm:
      name_ds = str(ds).split('(')[0]
      dynamic_algorithms_dict[name_ds] = ds
 
    performances = []
    vector_models = []
    for target in list(np.unique(bug_dataset['productName'])):
      
      # remove numbers
      target_name = ''.join(i for i in target if not i.isdigit())
      # initializing bad_chars_list
      bad_chars = [';', ':', '!', "*", " ", ".", ",", "-"]
      # using replace() to remove bad_chars
      for i in bad_chars:
        target_name = target_name.replace(i, '')

      test = bug_dataset.loc[bug_dataset['productName'] == target]
      test = test.select_dtypes(exclude=['object']).reset_index(drop=True)
      bugs = list(np.unique(test[test.columns[0]]))[1:]
      test[test.columns[0]] = test[test.columns[0]].replace(bugs, 1)
      
      train = bug_dataset[~bug_dataset['productName'].str.contains(target_name)]
      train = train.select_dtypes(exclude=['object']).reset_index(drop=True)
      bugs = list(np.unique(train[train.columns[0]]))[1:]
      train[train.columns[0]] = train[train.columns[0]].replace(bugs, 1)

      defective = np.count_nonzero(np.array(test[test.columns[0]]) == 1)

      percent_bugs =  round((defective / len(test)) * 100, 2)


      models_train, _ = self._Overproduction(train,
                                           dynamic_algorithms=dynamic_algorithms_dict,
                                           base_classifiers=base_classifiers_dict,
                                           size_pool=size_pool)
      vector_models.append(models_train)
      best_hyperparameter = self._selection(models_train, selection_approach)

      best_model, scaler = self._Overproduction(train,
                                           dynamic_algorithms=dynamic_algorithms_dict[best_hyperparameter.split('-')[0]],
                                           base_classifiers=base_classifiers_dict[best_hyperparameter.split('-')[1]],
                                           size_pool=int(best_hyperparameter.split('-')[2]),
                                           selected=True)

      evaluations = self._model_evaluating(test,
                                           best_model,
                                           scaler)  
     
      evaluations.insert(0, target)
      evaluations.insert(1, percent_bugs)
      evaluations.insert(2, best_hyperparameter.split('-')[0])
      evaluations.insert(3, best_hyperparameter.split('-')[1])
      evaluations.insert(4, best_hyperparameter.split('-')[2])

      cols = ['productName', 'percentBugs', 'dynamic_selection', 'classifier', 'size_pool', 'fscore', 'auc', 'pf', 'gmean', 'precision', 'recall', 'accuracy', 'tn', 'fp', 'fn', 'tp']
      performances.append(DataFrame([evaluations], columns=cols))
    self.vector_models = pd.concat(vector_models).reset_index(drop=True)
    self.performances = pd.concat(performances).reset_index(drop=True)

  def _model_evaluating(self, test, model, scaler):
    

    X_test = test.drop(test.columns[0], axis=1)
    y_test = test[test.columns[0]]

    bugs = list(np.unique(y_test))[1:]
    y_test = y_test.replace(bugs, 1)
    
    X_test = scaler.transform(X_test)
  
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    fscore = round(f1_score(y_test, y_pred), 5)
    auc = round(roc_auc_score(y_test, y_prob), 5)
    pf = round(fp / (fp + tn), 5)

    gmean = geometric_mean_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    return [fscore, auc, pf, gmean, precision, recall, accuracy, tn, fp, fn, tp]
  
  def _Overproduction(self, train,
                      dynamic_algorithms=None,
                      base_classifiers=None,
                      size_pool=None,
                      selected=False):
    """ Selecting the best model to predict the target project

    Parameters
    ----------
    DS: object (Default = None)
        The dynamic selection technique to fit and predict the target project.
        
        If None, then the dynamic selection technique is a
        :class:`~deslib.des.KNORAU`.

    base_estimator : object or list of base estimatos (Default = None)
        The base estimator used to generated the pool of classifiers. The base
        base_estimator should support the technique "predict_proba".
        
        If None, then the base estimator is a :class:`GaussianNB` from sklearn
        available on :class:`~sklearn.naive_bayes.GaussianNB`.

    Returns
      -------
      evaluations : list
          list with the evaluation values of measures.
    
    """
 
    if selected == False:
      result = []
      
      for name_ds in dynamic_algorithms:
        for name_clf in base_classifiers:
          for m in size_pool:
      
            X = train.drop(train.columns[0], axis=1)
            y = train[train.columns[0]]
            bugs = list(np.unique(y))[1:]
            y = y.replace(bugs, 1)
            scaler =  StandardScaler()
            X = scaler.fit_transform(X)

            X_train, X_dsel, y_train, y_dsel = X, X, y, y
            pool_classifiers = BaggingClassifier(base_estimator=base_classifiers[name_clf], n_estimators=m)
            pool_classifiers.fit(X_train, y_train)
            model = dynamic_algorithms[name_ds].set_params(pool_classifiers=pool_classifiers)
            model.fit(X_dsel, y_dsel)

            X_train = scaler.transform(X_train)
            y_score = model.predict_proba(X_train)[:, 1]
            y_pred = model.predict(X_train)
            tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()
            f1 = f1_score(y_train, y_pred)
            auc = roc_auc_score(y_train, y_score)
            pf = round(fp / (fp + tn), 5)

            gmean = geometric_mean_score(y_train, y_pred)
            precision = precision_score(y_train, y_pred)
            recall = recall_score(y_train, y_pred)
            acc = accuracy_score(y_train, y_pred)

            values = [name_ds, name_clf, m, f1, auc, pf, gmean, precision, recall, acc, tn, fp, fn, tp]
            result.append(pd.DataFrame([values], columns=['dynamic_selection', 'classifier', 'size_pool', 'fscore', 'auc', 'pf', 'gmean', 'precision', 'recall', 'accuracy', 'tn', 'fp', 'fn', 'tp']))

      vector_models = pd.concat(result).reset_index(drop=True)
      return vector_models, scaler

    else:
      X = train.drop(train.columns[0], axis=1)
      y = train[train.columns[0]]
      bugs = list(np.unique(y))[1:]
      y = y.replace(bugs, 1)
      scaler =  StandardScaler()
      X = scaler.fit_transform(X)

      X_train, X_dsel, y_train, y_dsel = X, X, y, y
      pool_classifiers = BaggingClassifier(base_estimator=base_classifiers, n_estimators=size_pool)
      pool_classifiers.fit(X_train, y_train)
      model = dynamic_algorithms.set_params(pool_classifiers=pool_classifiers)
      model.fit(X_dsel, y_dsel)
      return model, scaler

  def _selection (self, result, selection_approach):

    if selection_approach == 'rank':
      ds_list = list(np.unique(result['dynamic_selection']))
      clf_list = list(np.unique(result['classifier']))
      size_pool_list = list(np.unique(result['size_pool']))

      combination = []
      for i in ds_list:
        for j in clf_list:
          for k in size_pool_list:
            combination.append(i+'-'+j+'-'+str(k))

      for rowIndex in result.index:
        index_values = list(result.loc[rowIndex])
    
        result.loc[rowIndex, 'Model'] = index_values[0]+'-'+index_values[1]+'-'+str(index_values[2])

      values = result[['fscore', 'auc', 'pf']]
      
      algorithms_names = combination
      values = values.transpose()
      
      rank_array = []
      for count, index in enumerate(values.index):
        performances_array = values.to_numpy()
        performances_array = [performances_array[count]]
        
        if index == 'pf':
          ranks = np.array([rankdata(p) for p in performances_array])
        else:
          ranks = np.array([rankdata(-p) for p in performances_array])  
        
        rank_array.append(ranks)

      average_ranks = np.mean(rank_array, axis=0)
      geral_avg = pd.DataFrame(average_ranks, columns=[algorithms_names]).transpose().reset_index()
      geral_avg.columns = ['dynamic_selection', 'Rank']

      geral_avg = geral_avg.sort_values(by='Rank').reset_index(drop=True)
    
      min_avg = geral_avg['Rank'].min()
    
      best_hyperparameter = geral_avg.loc[geral_avg['Rank'] == min_avg, 'dynamic_selection'].to_numpy()[0]

      return best_hyperparameter

    elif selection_approach == 'fscore': 

      best = result.loc[result['fscore'] == max(list(result['fscore']))]
      ds = best['dynamic_selection'].values[0]
      clf = best['classifier'].values[0]
      size_pool = str(best['size_pool'].values[0])
      best_hyperparameter = ds+'-'+clf+'-'+size_pool
      return best_hyperparameter

    elif selection_approach == 'auc': 

      best = result.loc[result['auc'] == max(list(result['auc']))]
      ds = best['dynamic_selection'].values[0]
      clf = best['classifier'].values[0]
      size_pool = str(best['size_pool'].values[0])
      best_hyperparameter = ds+'-'+clf+'-'+size_pool

    else:
      best = result.loc[result['pf'] == min(list(result['pf']))]
      ds = best['dynamic_selection'].values[0]
      clf = best['classifier'].values[0]
      size_pool = str(best['size_pool'].values[0])

      best_hyperparameter = ds+'-'+clf+'-'+size_pool
      return best_hyperparameter
