from setuptools import setup
from setuptools import find_packages


setup(name='Checkers',
      version='0.1',
      description='Forecasting functions',
      url='',
      author='Felix Kleine Bösing',
      author_email='felix.boesing@t-online.de',
      license='FKB',
      packages=find_packages(),
      install_requires=['numpy', 'pandas', 'xgboost', 'recordclass', 'catboost', 'isoweek', 'statsmodels', 'lightgbm',
                        'sklearn', 'aenum', 'redis', 'flask'],
      zip_safe=False)