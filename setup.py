from setuptools import setup, find_packages

setup(
    name="boson-mlops-code",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "kfp==2.11.0",
        "google-cloud-aiplatform==1.71.1",
    ],
)
