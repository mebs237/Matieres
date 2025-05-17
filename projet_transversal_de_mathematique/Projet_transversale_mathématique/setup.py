from setuptools import setup, find_packages

setup(
    name="rangeland_dynamics",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
    ],
    author="Prestonne Mebou",
    description="Modélisation de la dynamique des pâturages",
)