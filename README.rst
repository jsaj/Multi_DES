DSSC
--------------

DSSC é um método supervisionado que utilizada técnicas de seleção dinâmica para realizar a predição de defeitos entre projetos.
Este método é centrado em técnicas da biblioteca **DESlib** [1]_, bem como algoritmos de aprendizagem de máquina disponíveis na API scikit-learn_ [2]_. 

Internamente, o DSSC requer que alguns processos sejam realizados antes do treinamento e predição. O DSSC requer o seguinte:

1. Local onde os dados do experimento são armazenados
2. Pré-processamentos dos dados devem seguir uma definição preestabelecida
 * Dado que este método busca predizer se um determinado projeto é ou não defeituoso, o processo de predição requer que os dados possuam apenas dois rótulos, defeito e não defeito, ou seja, opera somente com dados binários. Além disso, requer que as algumas *features* (colunas) dos dados sigam uma sequência predefinida, isto é, *bug label* e LOC (linha de código – *Lines of Code*) na primeira e segunda coluna, respectivamente. Para mais detalhes, verificar a página example_
3. O treinamento e avaliação de modelos de previsão
 * É possível treinar e avaliar várias técnicas de seleção dinâmica, bem como utilizar diversos algoritmos de aprendizagem de máquina. Então, todos os modelos gerados usam as mesmas etapas de processamento de dados, treinamento e avaliação

Como funciona?
--------------

O DSSC, considerando a natureza da previsão de defeitos entre projetos, é centrado em algumas etapas principais, tais como:

1. **Project Filtering phase**, cada projeto e suas *n* versões são verificadas se possuem o número mínimo de instâncias

2. Optimization Process phase
    1. **Target Definition**, cada projeto apto à predição é definido como conjunto de teste, enquanto os demais são atribuídos ao conjunto de treinamento seguindo o cenário *strict* CPDP [3]_
    2. **Model Generating**, realiza a geração e treinamento do modelo
    3. **Model Evaluating**, processo de avaliação do modelo com as medidas de desempenho sem reconhecimento de esforço

Um detalhe importante sobre a etapa de optimização é que, uma vez que o processamento, treinamento e avaliação são feitos, os resultados são tomados usando diferentes medidas de desempenho:

* Medidas de Desempenho sem Reconhecimento de Esforço (NPMs – *Non-effort-aware Performance Measures*)
    1. *F1-score*
    2. *Área sob a curva (AUC)*
    3. *Probabilidade de Alarme Falso (False Alarm - PF)*
    
* Medidas de Desempenho com Reconhecimento de Esforço (EPMs – *Effort-aware Performance Measures*)
    1. *IFA*
    2. *PII@20%*
    3. *PII@1000*
    4. *PII@2000*
    5. *CostEffort@20%*
    6. *CostEffort@1000*
    7. *CostEffort@2000*
    8. *Popt*

Os resultados são armazenados em arquivos CSV. Vale ressaltar que, o DSSC não realiza uma avaliação adicional dos resultados. Portanto, isso precisa ser criado por scripts externos; esta abordagem apenas realiza a geração de resultados usando diferentes configurações experimentais.

Requisitos:
-------------

O DSSC foi testado para funcionar com Python 3.5 or maior. Os requisitos de dependências são:

* scipy
* numpy
* scikit-learn
* deslib
* glob

Essas dependências são instaladas automaticamente usando os comandos pip abaixo.

Instalação
--------------

O DSSC requer, principalmente, do pacote deslib instalado usando o comando pip:

.. code-block:: bash

    pip install deslib
 
Além disso, o seguinte comando é necessário para utilizar o método DSSC:

.. code-block:: bash

    !git clone https://github.com/jsaj/dssc.git

Exemplo
--------------

Aqui, mostramos um exemplo do DSSC com suas configurações padrões:

.. code-block:: python

    from dssc.DSSC import DSSC
    import numpy as np
    import pandas as pd
    
    # dataset examples: AEEEM, NASA, PROMISE, RELINK
    dataset = '/content/dssc/Datasets/RELINK'

    # create object for defect prediction 
    model = DSSC(url_dataset=dataset)

    # calculates and optimizes results in relation to NPM and EPM
    npm, epm = model.optimization_process()

    print(npm, '\n\n', epm)

Além da predição com parâmetros padrões, o método DSSC aceita qualquer lista de técnicas de seleção dinâmica (do deslib) e lista de classificadores (do scikit-learn) como entrada, incluindo uma lista contendo diferentes métodos de preprocessamento (do scikit-learn). Mais exemplos para usar a API podem ser encontrados na página de example_.

# References
-----------
.. [1] : Rafael M. O. Cruz, Luiz G. Hafemann, Robert Sabourin and George D. C. Cavalcanti DESlib: A Dynamic ensemble selection library in Python. arXiv preprint arXiv:1802.04967 (2018).

.. [2] : F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay, “Scikit-learn: Machine learning in Python,” Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011.

.. [3] : S. Herbold, A. Trautsch, and J. Grabowski, “Global vs. local models for cross-project defect prediction,” Empirical software engineering, vol. 22, no. 4, pp. 1866–1902, 2017.

.. _scikit-learn: http://scikit-learn.org/stable/

.. _GitHub: https://github.com/scikit-learn-contrib/DESlib

.. _example: https://github.com/jsaj/dssc/blob/master/example.ipynb
