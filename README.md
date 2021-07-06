# sloop
Code repository to reproduce Spatial Language Object-Oriented POMDPs from [Spatial Language Understanding for Object Search in Partially Observed City-scale Environments](h2r.github.io/docs).

## Installation

The required python version is Python 3.6+.

1. Clone the repository and create and virtual environment with the following lines:
   ```
   git clone git@github.com:h2r/sloop.git
   cd sloop;
   virtualenv -p $(which python3) venv/sloop
   source venv/sloop/bin/activate
   ```

2. Then, install the dependencies in `requirements.txt`; The `>=` symbol in `requirements.txt` assumes backwards compatibility of those packages:
   ```
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. Install pomdp-py following [these instructions](https://h2r.github.io/pomdp-py/html/installation.html). The test version is `1.2.4.1`. You can specify that version when doing `pip install pomdp-py`.


4. Install the `sloop` package.

    ```
    # Assume you're at the root of the sloop repository
    pip install -e .
    ```




Finally, install the repository as a package. Assuming you are at the root directory of this repository,
```
pip install -e .
```

## Downloading Data
Run the following bash script from the root of the repository to download the SL-OSM (Spatial Language - Open Street Map) dataset in the correct location. This may take a minute or two as the file is around 4GB.

```
bash download-osm-data.sh
```

Next, we download the OO-POMDP resources. Again, running this script from the repo's root will allow you to place it in the right location.

```
bash download-oopomdp-data.sh
```

## Troubleshoot

1. Test map_info_dataset.py
    ```
    TODO
    ```

2. Test parsing
    ```
    TODO
    ```

3. Test filepaths
    ```
    TODO
    ```

4. Test interface

    Now, test the setup by running `interface.py`:
    ```
    cd spatial_foref/oopomdp/experiments
    python interface.py
    ```

## System Parameters
TODO: something about the extent of what the user can do with interface.py
