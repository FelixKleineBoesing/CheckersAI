from setuptools import setup
from setuptools import find_packages


setup(name='checkers',
      version='0.1',
      description='Checkers environment and Reinforcement Learning',
      url='',
      author='Felix Kleine Bösing',
      author_email='felix.boesing@t-online.de',
      license='MIT',
      packages=["checkers"],
      install_requires=['numpy', "pandas", 'tensorflow', 'keras'],
      zip_safe=False)