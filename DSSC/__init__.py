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
