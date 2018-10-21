from setuptools import setup
from setuptools import find_packages


setup(name='wdlforecast',
      version='0.1',
      description='Forecasting functions',
      url='https://172.31.175.49:30000/marcuscramer/wdl-python',
      author='Marcus Cramer',
      author_email='cramer@westphalia-datalab.com',
      license='WDL',
      packages=find_packages(),
      install_requires=['numpy', 'pandas', 'xgboost', 'recordclass', 'catboost', 'isoweek', 'statsmodels', 'lightgbm',
                        'sklearn', 'aenum', 'redis', 'flask'],
      zip_safe=False)