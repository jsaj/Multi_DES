import setuptools

# Reads the content of your README.md into a variable to be used in the setup below
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='dssc',  # should match the package folder
    packages=['dssc'],  # should match the package folder
    version='0.0.1',  # important for updates
    license='MIT',  # should match your chosen license
    maintainer='Juscelino S. A. Júnior',
    maintainer_email='j.jr.avelino@gmail.com',
    description='Implementation of approach to Cross-Project Defect Prediction with Dynamic Ensemble Selection methods',
    long_description=long_description,  # loads your README.md
    long_description_content_type="text/markdown",  # README.md is of type 'markdown'
    author='Juscelino S. A. Júnior',
    author_email='j.jr.avelino@gmail.com',
    url='https://github.com/jsaj/DSSC',
    install_requires=[
          'scikit-learn>=0.21.0',
          'numpy>=1.17.0',
          'scipy>=1.4.0',
    ],
    
    keywords=["cpdp", "dynamic selection", "prediction", "classification", "defect prediction"],  # descriptive meta-data
    classifiers=[  # https://pypi.org/classifiers
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Documentation',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],

#     download_url="https://github.com/mike-huls/py-console/archive/refs/tags/0.1.4.tar.gz",
)
