# sloop
Code repository to reproduce Spatial Language Object-Oriented POMDPs from [Spatial Language Understanding for Object Search in Partially Observed City-scale Environments](h2r.github.io/docs).

## Installation
The required python version is Python 3.6+. Once you've installed the correct version, clone the repository and create and virtual environment with the following lines:
```
git clone git@github.com:h2r/spatial-lang.git
virtualenv -p $(which python3) sloop
source sloop/bin/activate
```

Then, install the dependencies:
```
pip install --upgrade pip
pip install -r requirements.txt
```

Install [pomdp-py](https://h2r.github.io/pomdp-py/html/installation.html) by following the steps in the linked website. Following these steps is important for the POMDP to work.


Finally, install the repository as a package. Assuming you are at the root directory of this repository,
```
pip install -e .
```

## Downloading Data

## Training