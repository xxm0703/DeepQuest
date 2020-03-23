from setuptools import setup, find_packages

with open("VERSION", "r") as fv:
    VERSION = fv.read()

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="DeepQuest",
    version=VERSION,
    author="Martin Jordanov",
    author_email="martistj@gmail.com",
    description="A Machine Learning Algorithm destined to play Atari games",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xxm0703/DeepQuest",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy",
        "matplotlib",
        "tensorflow",
        "keras",
        "gym[atari]",
    ],
)
