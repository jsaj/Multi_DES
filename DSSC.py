# coding=utf-8

# Author: Juscelino S. Avelino Júnior <j.jr.avelino@gmail.com>
#
# License: BSD 3 clause

from glob import glob
import pandas as pd
import numpy as np

from warnings import filterwarnings
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# base classifiers
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import f1_score, auc, roc_auc_score
from scipy import *

# from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

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


  def __init__(self, url_dataset, with_PF=True):

    dataset_name = url_dataset.split('/')
    self.dataset_name = dataset_name[len(dataset_name)-1]
    self.url_dataset = url_dataset + '/*'
    # self.pf_has_been_called = False

    dataset_total = []

    for project in glob(self.url_dataset):
      project_name = project.split('/')
      project_name = project_name[len(project_name)-1].split('.csv')[0]
      ds = pd.read_csv(project)

      if 'name' in list(ds.columns):
        ds.pop('name')
        ds['name'] = project_name
      else:
        ds['name'] = project_name
      dataset_total.append(ds)
    
    dataset_total = pd.concat(dataset_total).reset_index(drop=True)

    #project filtering stage
    if with_PF == True:
      dataset_total = self._project_filtering(dataset_total)

    self.dataset_total = dataset_total
    self.train, self.test = [], []
    self.percent_bugs = 0
  
  def _project_filtering(self, dataset_total):

    """Filter of projects for prediction.

    Each project and its versions are checked to see if they have a minimum
    number of 5 instances of each label (defect and non-defect).
 
    Parameters
    ----------
    with_PF : Boolean (Default = False)
        Determines if the filter is applied to check if project have a minimum
        number of instances.

    Returns
    ----------
    dataset_total : DataFrame
        DataFrame containing all projects with minimum number of instances.

    References
    ----------

    S. Herbold,  A. Trautsch,  and J. Grabowski. "A comparative study to
    benchmark cross-project  defect  prediction  approaches". IEEE  Transactions
    on  Software  Engineering, vol. 44, no. 9, pp. 811–833, 2017.

    """
    return dataset_total

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

    test_data = dataset_total.loc[dataset_total['name'] == target_project]
    test_data = test_data.select_dtypes(exclude=['object']).reset_index(drop=True)
    bugs = list(np.unique(test_data[test_data.columns[0]]))[1:]
    test_data[test_data.columns[0]] = test_data[test_data.columns[0]].replace(bugs, 1)
    
    train_data = dataset_total[~dataset_total['name'].str.contains(target_name)]
    for project_name in list(np.unique(train_data['name'])):
      ds = train_data.loc[train_data['name'] == project_name]
      y = ds[ds.columns[0]]
      bugs = list(np.unique(y))[1:]
      y = y.replace(bugs, 1)
      defective = round((np.count_nonzero(np.array(y) == 1) / len(y)) * 100, 2)
      no_defective = round((np.count_nonzero(np.array(y) == 0) / len(y)) * 100, 2)

      if defective < 5.0 or no_defective < 5.0 or len(ds) < 100:
        train_data = train_data[train_data['name'] != project_name]
    train_data = train_data.select_dtypes(exclude=['object']).reset_index(drop=True)
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
    bugs = list(np.unique(y))[1:]
    y = y.replace(bugs, 1)
    self.percent_bugs = round((np.count_nonzero(np.array(y) == 1) / len(y)) * 100, 2)

  def _calc_popt(self, defective, LOC, effort):
    
    effort_instances =  (effort * defective[LOC].sum())/100
    index = defective[LOC].cumsum().searchsorted(effort_instances)
    TargetList = defective[:index]
    y_test = TargetList[TargetList.columns[0]]
    bugs = list(np.unique(y_test))[1:]
    y_test = y_test.replace(bugs, 1) 
    effort_percent = np.arange(0, 101, 1)
    defective_list = []

    for percent in range(0, 101):
      effort_loc =  (percent * TargetList[LOC].sum())/100
      index = TargetList[LOC].cumsum().searchsorted(effort_loc)
      data = defective[:index]
      
      if len(data) != 0 and percent < 100:
        bugs = np.count_nonzero(data[data.columns[0]] == 1)
        percent_bugs = bugs / np.count_nonzero( y_test == 1)
        defective_list.append(percent_bugs)
      elif len(data) != 0 and percent == 100:
        percent_bugs = 1.0
        defective_list.append(percent_bugs)
      else:
        percent_bugs = 0.0
        defective_list.append(percent_bugs)
      
    x = effort_percent
    y = defective_list

    y_a = np.arange(0.0, 1.01, 0.02)
    x_b = np.arange(0, 101, 2)

    x_a, y_b, h = [], [], []
    for i in range(51):
      x_a.append(0)
      y_b.append(1)
      h.append(100)

    x1 = np.concatenate([x_a, x_b])
    y1  = np.concatenate([y_a, y_b])

    x2 = np.concatenate([x_b, h])
    y2 = np.concatenate([x_a, y_a])

    area_P_R = auc(x, y) - auc(np.arange(0, 101, 1), np.arange(0.00, 1.01, 0.01))
    area_O_P = auc(x1, y1) - auc(x, y)
    area_P_R = auc(x, y) - auc(np.arange(0, 101, 1), np.arange(0.00, 1.01, 0.01))
    area_R_W = auc(np.arange(0, 101, 1), np.arange(0.00, 1.01, 0.01)) - auc(x2, y2)

    popt = 1 - (area_O_P/ (area_O_P + area_P_R + area_R_W))
    if popt > 1:
      popt = 1.0
    return popt
  
  def _calf_IFA(self, defective, LOC, model, scaler, effort):
    X_test = defective.drop(defective.columns[0], axis=1)
    if scaler != None:
      X_test = scaler.transform(X_test)
    y_pred = model.predict(X_test)

    defective['predict'] = y_pred
    effort_instances =  (effort * defective[LOC].sum())/100
    index = defective[LOC].cumsum().searchsorted(effort_instances)
    TargetList = defective[:index]
    y_test = TargetList[TargetList.columns[0]]
    bugs = list(np.unique(y_test))[1:]
    y_test = y_test.replace(bugs, 1) 
    
    IFA = 0
    for pred, true in zip(y_pred, y_test):
      if true == 1 and pred == 1:
        break
      elif true == 0 and pred == 1:
        IFA +=1

    return IFA
  
  def _calc_PIIL_CEL(self, defective, LOC, model, scaler, effort):
    
    if 'predict' in list(defective.columns):
      defective = defective.drop('predict', axis=1)
    X_test = defective.drop(defective.columns[0], axis=1)

    y_test = defective[defective.columns[0]]
    bugs = list(np.unique(y_test))[1:]
    y_test = y_test.replace(bugs, 1)

    if scaler != None:
      X_test = scaler.transform(X_test)
    y_pred = model.predict(X_test)

    defective['predict'] = y_pred
    if effort == 20:
      effort_instances =  (effort * defective[LOC].sum())/100
      index = defective[LOC].cumsum().searchsorted(effort_instances)
    elif effort == 1000:
      index = defective[LOC].cumsum().searchsorted(effort)
    else:
      index = defective[LOC].cumsum().searchsorted(effort)
 
    TargetList = defective[:index]

    real_bugs = np.count_nonzero(y_test == 1)

    PII = np.count_nonzero(TargetList['predict'] == 1)/ len(defective)
    CE = np.count_nonzero(TargetList['predict'] == 1)/ real_bugs

    if PII > 1.0:
      PII = 1.0
    if CE > 1.0:
      CE = 1.0  
    return PII, CE
  
  def _model_evaluating(self, model, scaler):
    
    cols = list(self.test.columns)
    LOC = cols[1]

    X_test = self.test.drop(self.test.columns[0], axis=1)
    y_test = self.test[self.test.columns[0]]
    bugs = list(np.unique(y_test))[1:]
    y_test = y_test.replace(bugs, 1)
    
    if scaler != None:
      X_test = scaler.transform(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    test_data = self.test
    test_data['score'] = y_proba
    test_data['score*loc'] = test_data['score'] * test_data[LOC]
    defective = test_data.loc[test_data[test_data.columns[0]] == 1]
    defective = defective.sort_values(by='score*loc', ascending=False)
    no_defective = test_data.loc[test_data[test_data.columns[0]] == 0]
    no_defective = no_defective.sort_values(by='score*loc', ascending=False)

    DEFECTIVE = pd.concat([defective, no_defective]).reset_index(drop=True)
    DEFECTIVE = DEFECTIVE.replace({inf: 1.0})

    X_test = DEFECTIVE.drop([DEFECTIVE.columns[0], 'score*loc', 'score'], axis=1)
    y_test = DEFECTIVE[DEFECTIVE.columns[0]]

    bugs = list(np.unique(y_test))[1:]
    y_test = y_test.replace(bugs, 1)

    if scaler != None:
      X_test = scaler.transform(X_test)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    F1 = round(f1_score(y_test, y_pred), 5)
    AUC = round(roc_auc_score(y_test, y_prob), 5)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    PF = round(fp / (fp + tn), 5)

    NPM = [F1, AUC, PF, precision, recall, accuracy]

    test_data = test_data.drop('score*loc', axis=1)

    test_data['score/loc'] = test_data['score'] / test_data[LOC]
    defective = test_data.loc[test_data[test_data.columns[0]] == 1]
    defective = defective.sort_values(by='score/loc', ascending=False)
    no_defective = test_data.loc[test_data[test_data.columns[0]] == 0]
    no_defective = no_defective.sort_values(by='score/loc', ascending=False)
    
    DEFECTIVE = pd.concat([defective, no_defective]).reset_index(drop=True)
    DEFECTIVE = DEFECTIVE.replace({inf: 1.0})

    DEFECTIVE = DEFECTIVE.drop(['score/loc', 'score'], axis=1)

    IFA =  self._calf_IFA(DEFECTIVE, LOC, model, scaler, 20)
    PII20, CE20 = self._calc_PIIL_CEL(DEFECTIVE, LOC, model, scaler, 20)
    PII1000, CE1000 = self._calc_PIIL_CEL(DEFECTIVE, LOC, model, scaler, 1000)
    PII2000, CE2000 = self._calc_PIIL_CEL(DEFECTIVE, LOC, model, scaler, 2000)
    Popt = self._calc_popt(DEFECTIVE, LOC, 20) 
    EPM = [IFA, PII20, PII1000, PII2000, CE20, CE1000, CE2000, Popt]

    return NPM, EPM
  
  def _model_building(self, ds,
                     base_estimator,
                     scaler,
                     resample_strategy,
                     dsel_size):
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

    scaler: object or list of scaler algorithms (Default = None)
        The scaler algorithm to transform features by scaling each
        feature to a given range.

    resample_strategy : {'over', 'under', None} (Default = None)
        The algorithm to perform random sampling

        - 'over' will use :class:`RandomOverSampler` from imblearn
          available on :class:`~imblearn.underesample_strategyampling.RandomOverSampler`

        - 'under' will use :class:`RandomUnderSampler` from imblearn
          available on :class:`~imblearn.underesample_strategyampling.RandomUnderSampler`

        - None, will not use algorithm to perform random sampling.   

    Returns
      -------
      NPM : list
          list with the array_npm values of non-effort-aware array_npm
          measures.
    
    """
    train_data = self.train
    X = train_data.drop(train_data.columns[0], axis=1)
    y = train_data[train_data.columns[0]]

    bugs = list(np.unique(y))[1:]
    y = y.replace(bugs, 1)

    if scaler != None:
      X = scaler.fit_transform(X)

    if dsel_size != None:
      X_train, X_dsel, y_train, y_dsel = train_test_split(X, y, test_size=dsel_size)
    else:
      X_train, y_train = X, y
      X_dsel, y_dsel = X, y 

    if resample_strategy not in ['over', 'smote', None]:
      raise ValueError("Value input is incorrect. Accept only three values: {'over', 'under', None}.")
                        
    if resample_strategy == 'over':
      resample_strategy = RandomOverSampler()
      X_train, y_train = resample_strategy.fit_resample(X_train, y_train)
      X_dsel, y_dsel = resample_strategy.fit_resample(X_dsel, y_dsel)

    elif resample_strategy == 'smote':
      resample_strategy = SMOTE()
      X_train, y_train = resample_strategy.fit_resample(X_train, y_train)
      X_dsel, y_dsel = resample_strategy.fit_resample(X_dsel, y_dsel)

    if base_estimator == None:
      base_estimator = GaussianNB()
    pool_classifiers = BaggingClassifier(base_estimator=base_estimator)
    pool_classifiers.fit(X_train, y_train)
    model = ds.set_params(pool_classifiers=pool_classifiers)
    model.fit(X_dsel, y_dsel)
    
    return model, scaler
  
  def dynamic_prediction(self, dynamic_algorithm=None,
                         base_estimator=None, preprocessing=None,
                         resample_strategy=None,
                         dsel_size=None):
      
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

    preprocessing: object or list of scaler algorithms (Default = None)
        The scaler algorithm to transform features by scaling each
        feature to a given range.

    resample_strategy : {'over', 'under', None} (Default = None)
        The algorithm to perform random sampling

        - 'over' will use :class:`RandomOverSampler` from imblearn
          available on :class:`~imblearn.underesample_strategyampling.RandomOverSampler`

        - 'under' will use :class:`RandomUnderSampler` from imblearn
          available on :class:`~imblearn.underesample_strategyampling.RandomUnderSampler`

        - None, will not use algorithm to perform random sampling.

    dsel_size : float (Default = None)
        The strategy to division of training data into TRAIN and DSEL
        
        If float, should be between 0.2 and 0.5 and represent the proportion of
        the training dataset to include in the DSEL split. If None, TRAIN and
        DSEL will receive all instances of training data.

    Returns
    -------
    best_model : object
        The best model to predict the target project.

    best_scaler : object
        The best scaler algorithm to transform features by scaling.

    Note: if  best_scaler = None, then doesn't use any pre-processing algorithm.
    """ 

 

    if dynamic_algorithm == None:
      dynamic_algorithm = [KNORAU()]
    
    if type(base_estimator) != list and base_estimator != None:
      base_estimator = [base_estimator]
    elif base_estimator == None:
      base_estimator = [LogisticRegression(solver='liblinear')]

    if preprocessing == None:
      preprocessing = [preprocessing]
    elif type(preprocessing) != list and preprocessing != None:
      preprocessing = [preprocessing]

    if resample_strategy == None:
      resample_strategy = [resample_strategy]
    elif type(resample_strategy) != list and resample_strategy != None:
      resample_strategy = [resample_strategy]

    if (type(dsel_size) == float and dsel_size < 0.2) or (type(dsel_size) == float and dsel_size > 0.5):
      raise ValueError('Value inputed for dsel_size is invalid. Accepts only float between 0.2 and 0.5 or None.')
    elif dsel_size != None and type(dsel_size) != float:
      raise ValueError('Value inputed for dsel is invalid. Accepts only float between 0.2 and 0.5 or None.')

    NPM, EPM = [], []
    projetos_preditos = ['szybkafucha', 'termoproject', 'tomcat', 'velocity-1.4', 'velocity-1.5', 'velocity-1.6', 'workflow', 'wspomaganiepi']
    list_projects = list(np.unique(self.dataset_total['name']))
    print(list_projects)
    aux = []
    performance_NPM, performance_EPM, = [], []
    for target_project in list_projects:
      
      if target_project in projetos_preditos:
        print(target_project)
        for ds in dynamic_algorithm:
          string_ds = str(type(ds))
          string_ds = string_ds.split("'")[1]
          string_ds = string_ds.split(".")

          name_ds = string_ds[len(string_ds)-1]
          print(name_ds)
          if 'deslib' not in string_ds:
            raise ValueError('Input dynamic selection technique invalid!')
          epm_data = []
          for classifier in base_estimator:
            for s in preprocessing:
              for resampling in resample_strategy:
                self._target_definition(target_project, self.dataset_total)
                model, scaler = self._model_building(ds=ds, base_estimator=classifier,
                                                    scaler=s, resample_strategy=None,
                                                    dsel_size=dsel_size)
                array_npm, array_epm = self._model_evaluating(model=model, scaler=scaler)
                
                array_npm.insert(0, self.dataset_name)
                array_npm.insert(1, target_project)
                array_npm.insert(2, self.percent_bugs)
                array_npm.insert(3, name_ds)
                array_npm.insert(4, scaler)
        
                cols = ['Dataset', 'Project', 'Percent_Bugs', 'DS', 'scaler',  'f1', 'auc', 'pf']
                array_npm = pd.DataFrame([array_npm], columns=cols)         
                performance_NPM.append(array_npm)
                ds_data.append(array_npm)
                aux.append(array_npm)

                array_epm.insert(0, self.dataset_name)
                array_epm.insert(1, target_project)
                array_epm.insert(2, self.percent_bugs)
                array_epm.insert(3, name_ds)
                cols = ['Dataset', 'Project', 'Percent_Bugs', 'DS','IFA', 'PII20', 'PII1000', 'PII2000', 'CE20', 'CE1000', 'CE2000', 'Popt']
                array_epm = pd.DataFrame([array_epm], columns=cols) 
                epm_data.append(array_epm)
                performance_EPM.append(array_epm)
          epm_data = pd.concat(epm_data).reset_index(drop=True)
          
    performance_NPM = pd.concat(performance_NPM).sort_values(by='Percent_Bugs').reset_index(drop=True)
    dict_npm = dict()
    for metric in list(['f1', 'auc', 'pf']):
      array = []
      project_bugs = []
      for project in list(np.unique(performance_NPM['Project'])):
        p_data = performance_NPM.loc[performance_NPM['Project'] == project]
        metric_data = p_data[metric]
        if metric == 'pf':
          max = np.min(metric_data)
        else:
          max = np.max(metric_data)  
        array.append(max)
        aux = [project, list(np.unique(p_data['Percent_Bugs']))[0]]
        project_bugs.append(pd.DataFrame([aux], columns=['Project', '%']))
      dict_npm[metric] = array
    project_bugs = pd.concat(project_bugs).reset_index(drop=True)

    f1 = pd.DataFrame(dict_npm['f1'], columns=['f1'])
    auc = pd.DataFrame(dict_npm['auc'], columns=['auc'])
    pf = pd.DataFrame(dict_npm['pf'], columns=['pf'])

    NPM = pd.concat([f1, auc, pf], axis=1)
    NPM = pd.concat([project_bugs, NPM], axis=1).reindex(project_bugs.index)
  
    performance_EPM = pd.concat(performance_EPM).reset_index(drop=True)

    dict_epm = dict()
    for metric in list(['IFA', 'PII20', 'PII1000', 'PII2000', 'CE20', 'CE1000', 'CE2000', 'Popt']):
      array = []
      project_bugs = []
      for project in list(np.unique(performance_EPM['Project'])):
        p_data = performance_EPM.loc[performance_EPM['Project'] == project]
        metric_data = p_data[metric]
        if metric in ['IFA', 'PII20', 'PII1000', 'PII2000']:
          max = np.min(metric_data)
        else:
          max = np.max(metric_data)  
        array.append(max)
        aux = [project, list(np.unique(p_data['Percent_Bugs']))[0]]
        project_bugs.append(pd.DataFrame([aux], columns=['Project', '%']))
      dict_epm[metric] = array
    project_bugs = pd.concat(project_bugs).reset_index(drop=True)

    IFA = pd.DataFrame(dict_epm['IFA'], columns=['IFA']).reset_index(drop=True)
    PII20 = pd.DataFrame(dict_epm['PII20'], columns=['PII20']).reset_index(drop=True)
    PII1000 = pd.DataFrame(dict_epm['PII1000'], columns=['PII1000']).reset_index(drop=True)
    PII2000 = pd.DataFrame(dict_epm['PII2000'], columns=['PII2000']).reset_index(drop=True)
    CE20 = pd.DataFrame(dict_epm['CE20'], columns=['CE20']).reset_index(drop=True)
    CE1000 = pd.DataFrame(dict_epm['CE1000'], columns=['CE1000']).reset_index(drop=True)
    CE2000 = pd.DataFrame(dict_epm['CE2000'], columns=['CE2000']).reset_index(drop=True)
    Popt = pd.DataFrame(dict_epm['Popt'], columns=['Popt']).reset_index(drop=True)

    EPM = pd.concat([IFA, PII20, PII1000, PII2000, CE20, CE1000, CE2000, Popt], axis=1).reindex(project_bugs.index)
    EPM = pd.concat([project_bugs, EPM], axis=1).reindex(project_bugs.index)

    return NPM, EPM
