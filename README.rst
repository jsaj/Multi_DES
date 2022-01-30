DSSC
--------------
DSSC is a supervised method that uses dynamic selection techniques to perform cross-project defect prediction. This method is centered on techniques from the DESlib_[1]_ library, as well as machine learning algorithms available in the scikit-learn_[2]_ API.

Internally, DSSC requires some processes to be performed before training and prediction. DSSC requires the following:

1. Location where experiment data is stored;
2. Data pre-processing must follow a pre-established definition
 * Since this method seeks to predict whether a given project is defective or not, the prediction process requires that the data have only two labels, defect and non-defect, i.e, it operates only with binary data. Also, feature *bug label* must be in the first column. For more details, check out the example_ page
 
3. Training and evaluation
 * It is possible to use different parameters: dynamic selection techniques, different machine learning algorithms and different variations in the size of the *pool* for classifiers. So, all generated models use the same data processing, training and evaluation steps.
 
How it works?
--------------

The DSSC, considering the nature of cross-project defect prediction, is centered on a few key steps, such as:

1. **Target Definition**, each project is defined as a test set, while the others are assigned to the training set following the *strict* CPDP scenario [3]_
2. **Overproduction**, consists of defining a competent predictive model by training set to classify the test data.
3. **Model Evaluating**, model evaluation process with performance evaluation metricsnho.

* Performance evaluation metrics

     1. *F1-score*
     2. *Area under the curve ROC (ROC-AUC)*
     3. *False Alarm Probability (PF)*
    

Os resultados são armazenados em arquivos CSV. Vale ressaltar que, o DSSC não realiza uma avaliação adicional dos resultados. Portanto, isso precisa ser criado por scripts externos; esta abordagem apenas realiza a geração de resultados usando diferentes configurações experimentais.

Requisitos:
-------------

O DSSC foi testado para funcionar com Python 3.5 ou maior. Os requisitos são:

* scipy
* numpy
* scikit-learn
* deslib
* glob

Essas dependências são instaladas automaticamente usando os comandos pip abaixo.

Instalação
--------------
 
Além disso, o seguinte comando é necessário para utilizar o método DSSC:

.. code-block:: bash

    git clone https://github.com/jsaj/dssc.git

Exemplo
--------------

Aqui, mostramos um exemplo do DSSC com suas configurações padrões:

.. code-block:: python

    from dssc.DSSC import DSSC
    import numpy as np
    import pandas as pd
    
    # dataset examples: AEEEM, NASA, PROMISE, RELINK
    dataset = '/content/dssc/Datasets/RELINK'
    
    # directory to save results
    save_directory = '/content/sample_data/Results'
    
    # create object for defect prediction 
    dssc_obj = DSSC(url_dataset=dataset, save_directory=save_directory)

    # calculates and optimizes results in relation to NPM and EPM
    npm, epm = dssc_obj.optimization_process(preprocessing=preprocessing)

    print(npm, '\n\n', epm)

Além da predição com parâmetros padrões, o método DSSC aceita qualquer lista de técnicas de seleção dinâmica (do deslib) e lista de classificadores (do scikit-learn) como entrada, incluindo uma lista contendo diferentes métodos de preprocessamento (do scikit-learn). Mais exemplos para usar a API podem ser encontrados na página de example_.

# Referências
-----------
.. [1] : Rafael M. O. Cruz, Luiz G. Hafemann, Robert Sabourin and George D. C. Cavalcanti DESlib: A Dynamic ensemble selection library in Python. arXiv preprint arXiv:1802.04967 (2018).

.. [2] : F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay, “Scikit-learn: Machine learning in Python,” Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011.

.. [3] : S. Herbold, A. Trautsch, and J. Grabowski, “Global vs. local models for cross-project defect prediction,” Empirical software engineering, vol. 22, no. 4, pp. 1866–1902, 2017.

.. _scikit-learn: http://scikit-learn.org/stable/

.. _DESlib: https://github.com/scikit-learn-contrib/DESlib

.. _example: https://github.com/jsaj/dssc/blob/master/example.ipynb
