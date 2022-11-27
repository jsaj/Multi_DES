
MDS-CPDP
========

MDS-CPDP is a supervised method that uses dynamic selection techniques to perform cross-project defect prediction. This method is centered on techniques from the DESlib [1]_ library, as well as machine learning algorithms available in the scikit-learn_ API.

Internally, MDS-CPDP requires some processes to be performed before training and prediction. MDS-CPDP requires the following:

1. Location where experiment data is stored;
2. Data pre-processing must follow a pre-established definition
 * Since this method seeks to predict whether a given project is defective or not, the prediction process requires that the data have only two labels, defect and non-defect, i.e, it operates only with binary data. Also, feature *bug label* must be in the first column. For more details, check out the example_ page
 
3. Training and evaluation
 * It is possible to use different parameters: dynamic selection techniques, different machine learning algorithms and different variations in the size of the *pool* for classifiers. So, all generated models use the same data processing, training and evaluation steps.
 
How it works?
--------------

The MDS-CPDP, considering the nature of cross-project defect prediction, is centered on a few key steps, such as:

1. **Overproduction**, based on the training set, n models are generated using a series of parameters: dynamic selection techniques, base classifiers and sizes of pool of classifiers
2. **Selection**, consists of defining a competent predictive model by training set to classify the test data.
3. **Model Evaluating**, model evaluation process with performance evaluation metrics.

* Performance evaluation metrics

     1. *F1-score*
     2. *Area under the curve ROC (ROC-AUC)*
     3. *False Alarm Probability (PF)*
    

Results are stored in CSV files. It is worth mentioning that the MDS-CPDP does not carry out an additional evaluation of the results. So this needs to be created by external scripts; this approach only performs the generation of results using different experimental setups.

Requirements:
-------------

MDS-CPDP has been tested to work with Python 3.5 or greater. The requirements are:

* scipy(>=1.4.1)
* numpy(>=1.21.6)
* scikit-learn(>=1.0.2)
* deslib(>=0.3.5)
* glob(>=0.7)

These dependencies are automatically installed using the pip commands above.

Installation:
-------------

The package can be installed using:

Latest version (under development):

.. code-block:: bash

    !git clone https://github.com/jsaj/mds_cpdp.git

Also, need install deslib:

.. code-block:: bash

    !pip install deslib
    

Example
--------------

Here we show an example using the MDS-CPDP with default parameters.
We used the Google Colaboratory environment to run the experiments, so:

.. code-block:: python
    
    from MDS_CPDP.mdscpdp import MDSCPDP

    import pandas as pd
    from glob import glob

    import warnings
    warnings.filterwarnings("ignore")

    # path of datasets to predict
    path = '/content/MDS_CPDP/benchmark-execution/benchmarks/datasets/RELINK/*'

    # read and create dataframe (dataset) with all projects for predict
    dataset = []
    for project_url in glob(path):
      productName = project_url.split('/')[len(project_url.split('/'))-1]
      df = pd.read_csv(project_url)
      df.insert(0, 'productName', productName)
      dataset.append(df)
    dataset = pd.concat(dataset).reset_index(drop=True)

    #create MDSCPDP object to predict dataset
    obj = MDSCPDP(dataset)

    #get MDSCPDP performance after predict the dataset. Return a pandas dataframe
    obj.performances

In addition to prediction with default parameters, the MDS-CPDP method accepts any list of dynamic selection techniques (from deslib) and list of classifiers (from scikit-learn) as input, including a list containing different size for pool of classifier.

References:
-----------
.. [1] : Rafael M. O. Cruz, Luiz G. Hafemann, Robert Sabourin and George D. C. Cavalcanti DESlib: A Dynamic ensemble selection library in Python. arXiv preprint arXiv:1802.04967 (2018).

.. [2] : F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay, “Scikit-learn: Machine learning in Python,” Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011.

.. [3] : S. Herbold, A. Trautsch, and J. Grabowski, “Global vs. local models for cross-project defect prediction,” Empirical software engineering, vol. 22, no. 4, pp. 1866–1902, 2017.

.. _scikit-learn: http://scikit-learn.org/stable/

.. _DESlib: https://github.com/scikit-learn-contrib/DESlib

.. _example: https://github.com/jsaj/MDS_CPDP/blob/master/examples/example_base.ipynb
