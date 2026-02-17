from setuptools import setup, find_packages

setup(
    name="meta-learning-project",
    version="1.0.0",
    author="Your Name",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        'numpy>=1.24.0',
        'pandas>=2.0.0',
        'scikit-learn>=1.3.0',
        'openml>=0.14.0',
        'pymfe>=0.4.0',
        'matplotlib>=3.7.0',
        'seaborn>=0.12.0',
    ],
)