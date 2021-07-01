# sloop
Code repository to reproduce Spatial Language Object-Oriented POMDPs from [Spatial Language Understanding for Object Search in Partially Observed City-scale Environments](h2r.github.io/docs).

## Installation
The required python version is Python 3.6+. Once you've installed the correct version, clone the repository and create and virtual environment with the following lines:
```
git clone git@github.com:h2r/sloop.git
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
Run the following bash script from the root of the repository to download the SL-OSM (Spatial Language - Open Street Map) dataset in the correct location. This may take a minute or two as the file is around 4GB.

```
bash download-osm-data.sh
```

## Troubleshoot

Now, test the setup by running `interface.py`:
```
cd spatial_foref/oopomdp/experiments
python interface.py
```

## System Parameters
TODO: something along the lines of what the user can do with interface.py
