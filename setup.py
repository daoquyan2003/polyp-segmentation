from setuptools import find_packages, setup

setup(
    name="src",
    version="0.1.0",
    description=(
        "Polyp segmentation training pipeline"
        "based on pytorch_lightning and hydra"
    ),
    author="Quy-An Dao",
    author_email="daoquyan26122003@gmail.com",
    install_requires=["pytorch-lightning", "hydra-core"],
    packages=find_packages(exclude=["tests"]),
)
