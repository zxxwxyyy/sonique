from setuptools import setup, find_packages
import os 

with open('requirements.txt') as f:
    required_files = f.read().splitlines()
setup(
    name='efficient-video-bgm',
    version='0.0.1',
    description='Inference tools for Efficient-Video-BGM-Generation',
    packages=find_packages(),
    install_requires = required_files  
)