from setuptools import setup, find_packages

setup(
    name='IFCB_PROJECT',
    version='0.1',
    packages=find_packages('src'),
    package_dir={'': 'src'},
)