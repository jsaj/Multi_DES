DSSC
--------------

DSSC é um método supervisionado que utilizada técnicas de seleção dinâmica para realizar a predição de defeitos entre projetos.
Este método é centrado em técnicas da biblioteca **DESlib** [1]_, bem como algoritmos de aprendizagem de máquina disponíveis na API scikit-learn_ [2]_. 

Internamente, o DSSC requer que alguns processos sejam realizados antes do treinamento e predição. O DSSC requer o seguinte:

1. Local onde os dados do experimento são armazenados
2. Pré-processamentos dos dados devem seguir uma definição preestabelecida
 * Dado que este método busca predizer se um determinado projeto é ou não defeituoso, o processo de predição requer que os dados possuam apenas dois rótulos, defeito e não defeito, ou seja, opera somente com dados binários. Além disso, requer que as algumas *features* (colunas) dos dados sigam uma sequência predefinida. Para mais detalhes, verificar a página exemplo_
3. O treinamento e avaliação de modelos de previsão
 * É possível treinar e avaliar várias técnicas de seleção dinâmica, bem como utilizar diversos algoritmos de aprendizagem de máquina. Então, todos os modelos gerados usam as mesmas etapas de processamento de dados, treinamento e avaliação

Como funciona?
--------------

O DSSC, considerando a natureza da previsão de defeitos entre projetos, é centrado em algumas etapas principais, tais como:

* **Project Filtering phase**, cada projeto e suas *n* versões são verificadas se possuem o número mínimo de instâncias

* Optimization Process phase
    1. **Target Definition**, cada projeto apto à predição é definido como conjunto de teste, enquanto os demais são atribuídos ao conjunto de treinamento seguindo o cenário *strict* CPDP [3]_
    2. **Model Generating**, realiza a geração e treinamento do modelo
    3. **Model Evaluating**, processo de avaliação do modelo com as medidas de desempenho sem reconhecimento de esforço


Dependencies:
-------------

DESlib is tested to work with Python 3.5, 3.6 and 3.7. The dependency requirements are:

* scipy(>=1.4.0)
* numpy(>=1.17.0)
* scikit-learn(>=0.20.0)

These dependencies are automatically installed using the pip commands above.

Exemplo
--------------
  
Here we present an example of the KNORA-E techniques using a random forest to generate the pool of classifiers:

.. code-block:: python

    from sklearn.ensemble import RandomForestClassifier
    from deslib.des.knora_e import KNORAE

    # Train a pool of 10 classifiers
    pool_classifiers = RandomForestClassifier(n_estimators=10)
    pool_classifiers.fit(X_train, y_train)

    # Initialize the DES model
    knorae = KNORAE(pool_classifiers)

    # Preprocess the Dynamic Selection dataset (DSEL)
    knorae.fit(X_dsel, y_dsel)

    # Predict new examples:
    knorae.predict(X_test)

The library accepts any list of classifiers (from scikit-learn) as input, including a list containing different classifier models (heterogeneous ensembles).
More examples to use the API can be found in the `examples page <auto_examples/index.html>`_.


Citation
==================

If you use DESLib in a scientific paper, please consider citing the following paper:

Rafael M. O. Cruz, Luiz G. Hafemann, Robert Sabourin and George D. C. Cavalcanti **DESlib: A Dynamic ensemble selection library in Python.** arXiv preprint arXiv:1802.04967 (2018).

.. code-block:: text

    @article{JMLR:v21:18-144,
        author  = {Rafael M. O. Cruz and Luiz G. Hafemann and Robert Sabourin and George D. C. Cavalcanti},
        title   = {DESlib: A Dynamic ensemble selection library in Python},
        journal = {Journal of Machine Learning Research},
        year    = {2020},
        volume  = {21},
        number  = {8},
        pages   = {1-5},
        url     = {http://jmlr.org/papers/v21/18-144.html}
    }


# References
-----------
.. [1] : Rafael M. O. Cruz, Luiz G. Hafemann, Robert Sabourin and George D. C. Cavalcanti DESlib: A Dynamic ensemble selection library in Python. arXiv preprint arXiv:1802.04967 (2018).

.. [2] : F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay, “Scikit-learn: Machine learning in Python,” Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011.

.. [3] : S. Herbold, A. Trautsch, and J. Grabowski, “Global vs. local models for cross-project defect prediction,” Empirical software engineering, vol. 22, no. 4, pp. 1866–1902, 2017.

.. [4] : R. M. O. Cruz, R. Sabourin, and G. D. Cavalcanti, “Dynamic classifier selection: Recent advances and perspectives,” Information Fusion, vol. 41, pp. 195 – 216, 2018.

.. [5] : A. S. Britto, R. Sabourin, L. E. S. de Oliveira, Dynamic selection of classifiers - A comprehensive review, Pattern Recognition 47 (11) (2014) 3665–3680.

.. _scikit-learn: http://scikit-learn.org/stable/

.. _GitHub: https://github.com/scikit-learn-contrib/DESlib

.. _exemplo: https://github.com/jsaj/ml/blob/master/example


