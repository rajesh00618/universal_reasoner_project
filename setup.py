from setuptools import setup, find_packages

setup(
    name="universal_reasoner",
    version="0.1.0",
    description="A Neuro-Symbolic Reasoner for ANY ML Model",
    author="Rajesh Gurugubelli",  # Your Name
    packages=find_packages(),
    install_requires=[
        "shap",
        "numpy",
        "pandas",
        "scikit-learn"
    ]
)