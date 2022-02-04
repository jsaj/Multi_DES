# coding=utf-8

# Author: Juscelino S. Avelino Júnior <j.jr.avelino@gmail.com>
#

from glob import glob
import pandas as pd
import numpy as np
import re

from warnings import filterwarnings

# base classifiers
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

from deslib.des import KNORAU

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, accuracy_score
from imblearn.metrics import geometric_mean_score

from scipy import *
from scipy.stats import rankdata

from sklearn.utils.testing import ignore_warnings
# from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import confusion_matrix

class DSSC():
  """Dynamic Selection Supervised Cross-project defect prediction (DSSC).

    Parameters
    ----------

  References
    ----------

    R. M. Cruz, L. G. Hafemann, R. Sabourin, and G. D. Cavalcanti.
    "Deslib:  A dynamicensemble selection library in python."
    Journal of Machine Learning Research, vol. 21, no. 8, pp. 1–5, 2020.
  
  """


  def __init__(self, dataset,
               with_PF = False,
               dynamic_algorithm=None,
               base_estimator=None):
    
    self.dynamic_algorithm = dynamic_algorithm
    self.base_estimator = base_estimator
    self.dataset = dataset
   
  def predict_dataset(self):
    
    if self.dynamic_algorithm == None:
      self.dynamic_algorithm = [KNORAU()]
    
    if type(self.base_estimator) != list and self.base_estimator != None:
      self.base_estimator = [self.base_estimator]
    elif self.base_estimator == None:
      self.base_estimator = [DecisionTreeClassifier()]

    self.train, self.test, self.target_project = [], [], None
    self.percent_bugs = 0
  
    performances_dataset = []
    for target in list(np.unique(dataset['productName'])):

      print(target)
      self.target_project = target
      self._target_definition(target, self.dataset)

      result = self._model_generation(dynamic_algorithm=self.dynamic_algorithm,
                                      base_estimator=self.base_estimator,
                                      preprocessing=StandardScaler())
      
      model, scaler = self._selection(result)

  
      performances = self._model_evaluating(model=model, scaler=scaler)
    
      paramns = model.get_params()
    
      name_ds = str(model)
      name_ds = name_ds.split("(")[0]

      name_clf = str(paramns['pool_classifiers__base_estimator'])
      name_clf = name_clf.split("(")[0]
      
      performances.insert(0, self.target_project)
      performances.insert(1, self.percent_bugs)
      performances.insert(2, name_ds)
      performances.insert(3, name_clf)

      cols = ['productName', 'defects (%)', 'dynamicSelection', 'classifier', 'fscore', 'auc', 'pf', 'gmean', 'precision', 'recall', 'accuracy', 'tn', 'fp', 'fn', 'tp']
      performances = pd.DataFrame([performances], columns=cols)
  
      performances_dataset.append(performances)

    performances_dataset = pd.concat(performances_dataset).reset_index(drop=True)
   
    return performances_dataset
   
  def _target_definition(self, target_project, dataset_total):
    """ Selecting the best model to predict the target project

    Parameters
    ----------
    
    target_project : string
        String with name of target project.

    dataset_total : DataFrame
        DataFrame containing all projects for prediction

    Returns
    -------
    train : {DataFrame} of shape (n_samples, n_features)
        The training input samples.

    test : {DataFrame} of shape (n_samples, n_features)
        The target project input samples.

    """
    target_name = target_project.split('.csv')[0]

    if '-' in target_name:
      target_name = target_project.split('-')[0]

    if '.' in target_name:
      target_name = target_project.split('.')[0]

    for character in target_name:
      if character.isdigit():
        target_name = list(re.findall(r'(\w+?)(\d+)', target_name)[0])
        target_name = target_name[0]
        break
    
    
    test_data = dataset_total.loc[dataset_total['productName'] == target_project]
    
    test_data = test_data.select_dtypes(exclude=['object']).reset_index(drop=True)
    bugs = list(np.unique(test_data[test_data.columns[0]]))[1:]
    test_data[test_data.columns[0]] = test_data[test_data.columns[0]].replace(bugs, 1)
    
    train_data = dataset_total[~dataset_total['productName'].str.contains(target_name)]

    train_data = train_data.select_dtypes(exclude=['object']).reset_index(drop=True)
    train_data = train_data.drop_duplicates().reset_index(drop=True)
    bugs = list(np.unique(train_data[train_data.columns[0]]))[1:]
    train_data[train_data.columns[0]] = train_data[train_data.columns[0]].replace(bugs, 1)

    def Diff(li1, li2):
      return list(set(li1) - set(li2)) + list(set(li2) - set(li1))

    diff_columns = Diff(list(train_data.columns), list(test_data.columns))
    
    if len(diff_columns) > 0:
      for col in diff_columns:
        if col in list(train_data.columns):
          train_data.pop(col)
        elif col in list(test_data.columns):
          test_data.pop(col)

    self.test = test_data
    self.train = train_data
    y = self.test[self.test.columns[0]]

    self.percent_bugs = round((np.count_nonzero(np.array(y) == 1) / len(y)) * 100, 2)

  def _model_evaluating(self, model, scaler):
    
    test_data = self.test

    X_test = test_data.drop(test_data.columns[0], axis=1)
    y_test = test_data[test_data.columns[0]]

    bugs = list(np.unique(y_test))[1:]
    y_test = y_test.replace(bugs, 1)
    
    if scaler != None:
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

    performances_dataset = [fscore, auc, pf, gmean, precision, recall, accuracy, tn, fp, fn, tp]
    return performances_dataset
  
  def _model_generation(self, dynamic_algorithm=None,
                      base_estimator=None,
                      preprocessing=None,
                      resample_strategy=None):
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
      
        If None, then the base estimator is a :class:`RandomForestClassifier` from sklearn
        available on :class:`~sklearn.ensemble.RandomForestClassifier`.


    Returns
      -------
      performances_dataset : list
          list with the performances values of different models.
    
    """

    self.classifiers_dict = dict()
    for count in range(len(base_estimator)):
      name_clf = str(base_estimator[count])
      name_clf = name_clf.split('(')[0]
      self.classifiers_dict[name_clf] = base_estimator[count]

    self.dynamic_algorithm_dict = dict()
    for count in range(len(dynamic_algorithm)):
      name_ds = str(dynamic_algorithm[count])
      name_ds = name_ds.split('(')[0]
      self.dynamic_algorithm_dict[name_ds] = dynamic_algorithm[count]

    result = []
    id_model = 0

    for name_ds in self.dynamic_algorithm_dict:
      for name_clf in self.classifiers_dict:
        # for m in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        for m in [10]:
          train_data = self.train
          cols = list(train_data.columns)
          X_train = train_data.drop(train_data.columns[0], axis=1)
          y_train = train_data[train_data.columns[0]]
          bugs = list(np.unique(y_train))[1:]
          y_train = y_train.replace(bugs, 1)
          scaler =  StandardScaler()
          X_train = scaler.fit_transform(X_train)

          # X_train, X_dsel, y_train, y_dsel = train_test_split(X, y, test_size=0.5, random_state=0)   
          pool_classifiers = BaggingClassifier(base_estimator=self.classifiers_dict[name_clf], n_estimators=m)
          pool_classifiers.fit(X_train, y_train)
          model = self.dynamic_algorithm_dict[name_ds].set_params(pool_classifiers=pool_classifiers)
          model.fit(X_train, y_train)

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

          name_model = 'DSSC-'+name_ds
          values = [name_ds, name_clf, m, f1, auc, pf, gmean, precision, recall, acc, tn, fp, fn, tp]
          result.append(pd.DataFrame([values], columns=['dynamicSelection', 'classifier', 'nEstimators', 'fscore', 'auc', 'pf', 'gmean', 'precision', 'recall', 'accuracy', 'tn', 'fp', 'fn', 'tp']))

    result = pd.concat(result).reset_index(drop=True)

    return result

  def _selection (self, result):
    
    DS = list(np.unique(result['dynamicSelection']))
    CLF = list(np.unique(result['classifier']))
    n_estimators = list(np.unique(result['nEstimators']))

    combination = []
    for i in DS:
      for j in CLF:
        for k in n_estimators:
          combination.append(i+'-'+j+'-'+str(k))

    win_tie = dict()
    for c in combination:
      win_tie[c] = 0      

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
    geral_avg.columns = ['DS', 'Rank']

    geral_avg = geral_avg.sort_values(by='Rank').reset_index(drop=True)
  
    min_avg = geral_avg['Rank'].min()
   
    best_configuration = geral_avg.loc[geral_avg['Rank'] == min_avg, 'DS'].to_numpy()[0]
   
    res_best = []

    ds_name = best_configuration.split('-')[0]
    name_clf = best_configuration.split('-')[1]
    estimators = int(best_configuration.split('-')[2])

    X = self.train.drop(self.train.columns[0], axis=1)
    y = self.train[self.train.columns[0]]

    scaler =  StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, y_train = X, y

    pool_classifiers = BaggingClassifier(base_estimator=self.classifiers_dict[name_clf], n_estimators=estimators)
    pool_classifiers.fit(X_train, y_train)
    model = self.dynamic_algorithm_dict[ds_name].set_params(pool_classifiers=pool_classifiers)
    model.fit(X_train, y_train)

    return model, scaler
