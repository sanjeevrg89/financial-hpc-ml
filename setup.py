from setuptools import setup, find_packages

setup(
    name='fsi-hpc-ml',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'ray[tune]',
        'pandas',
        'scikit-learn',
        'numpy',
        'tensorflow',
        'torch',
        'transformers',
        'pyyaml'
    ],
)