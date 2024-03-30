from setuptools import setup, find_packages
import os 

with open('requirements.txt') as f:
    required_files = f.read().splitlines()
setup(
    name='sonique',
    version='0.0.2',
    description='Inference tools for SONIQUE',
    packages=find_packages(),
    install_requires = required_files  
)