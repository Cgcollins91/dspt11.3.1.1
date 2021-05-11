"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""
# Always prefer setuptools over distutils
from setuptools import setup
setup(
    name='dspt11_lambda_chris_collins',  # Required
    version='0.0.1',  # Required
    author='DSPT11',  # Optional
    author_email='chris_collins@lambdastudents.com',  # Optional
    keywords='reddit',  # Optional
    packages=['mymodule'],  # Required
    python_requires='>=3.6, <4',
    install_requires=['pandas','numpy'],  # Optional
)