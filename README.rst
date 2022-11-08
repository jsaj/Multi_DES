
MDS-CPDP
========

MDS-CPDP is a supervised method that uses dynamic selection techniques to perform cross-project defect prediction. This method is centered on techniques from the DESlib [1]_ library, as well as machine learning algorithms available in the scikit-learn_ API.

Internally, MDS-CPDP requires some processes to be performed before training and prediction. DSSC requires the following:

1. Location where experiment data is stored;
2. Data pre-processing must follow a pre-established definition
 * Since this method seeks to predict whether a given project is defective or not, the prediction process requires that the data have only two labels, defect and non-defect, i.e, it operates only with binary data. Also, feature *bug label* must be in the first column. For more details, check out the example_ page
 
3. Training and evaluation
 * It is possible to use different parameters: dynamic selection techniques, different machine learning algorithms and different variations in the size of the *pool* for classifiers. So, all generated models use the same data processing, training and evaluation steps.
 
How it works?
--------------

The MDS-CPDP, considering the nature of cross-project defect prediction, is centered on a few key steps, such as:

1. **Target Definition**, each project is defined as test, while the others are assigned to the training set following the *strict* CPDP scenario [3]_
2. **Overproduction**, consists of defining a competent predictive model by training set to classify the test data.
3. **Model Evaluating**, model evaluation process with performance evaluation metricsnho.

* Performance evaluation metrics

     1. *F1-score*
     2. *Area under the curve ROC (ROC-AUC)*
     3. *False Alarm Probability (PF)*
    

Results are stored in CSV files. It is worth mentioning that the DSSC does not carry out an additional evaluation of the results. So this needs to be created by external scripts; this approach only performs the generation of results using different experimental setups.

Requirements:
-------------

DSSC has been tested to work with Python 3.5 or greater. The requirements are:

* scipy(>=1.4.1)
* numpy(>=1.21.6)
* scikit-learn(>=1.0.2)
* deslib(>=0.3.5)
* glob(>=0.7)

These dependencies are automatically installed using the pip commands above.

Installation:
-------------

The package can be installed using pip:

Stable version:

.. code-block:: bash

    pip install mdscpdp

Latest version (under development):

.. code-block:: bash

    git clone https://github.com/jsaj/mds_cpdp.git
    

Examples
--------------

Here we show an example using the MDS-CPDP with default parameters:

.. code-block:: python

    from mdscpdp.MDSCPDP import MDSCPDP
    import numpy as np
    import pandas as pd
    
    # dataset examples: AEEEM, NASA, PROMISE, RELINK
    dataset = '/content/dssc/Datasets/RELINK'
    
    # directory to save results
    save_directory = '/content/sample_data/Results'
    
    # create object for defect prediction 
    dssc_obj = MDSCPDP(url_dataset=dataset, save_directory=save_directory)

    # calculates and optimizes results in relation to NPM and EPM
    npm, epm = dssc_obj.optimization_process(preprocessing=preprocessing)

    print(npm, '\n\n', epm)

In addition to prediction with default parameters, the DSSC method accepts any list of dynamic selection techniques (from deslib) and list of classifiers (from scikit-learn) as input, including a list containing different preprocessing methods (from scikit-learn). More examples for using the API can be found on the example_ page.

References:
-----------
.. [1] : Rafael M. O. Cruz, Luiz G. Hafemann, Robert Sabourin and George D. C. Cavalcanti DESlib: A Dynamic ensemble selection library in Python. arXiv preprint arXiv:1802.04967 (2018).

.. [2] : F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay, “Scikit-learn: Machine learning in Python,” Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011.

.. [3] : S. Herbold, A. Trautsch, and J. Grabowski, “Global vs. local models for cross-project defect prediction,” Empirical software engineering, vol. 22, no. 4, pp. 1866–1902, 2017.

.. _scikit-learn: http://scikit-learn.org/stable/

.. _DESlib: https://github.com/scikit-learn-contrib/DESlib

.. _example: https://github.com/jsaj/dssc/blob/master/example.ipynb
