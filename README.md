# sloop
Code repository to reproduce Spatial Language Object-Oriented POMDPs from [Spatial Language Understanding for Object Search in Partially Observed City-scale Environments](h2r.github.io/docs).

* [Installation](#installation)
* [Download Dataset & Models](#dataset-and-models)
* [Download Results & Reproduce](#results)
* [OpenStreetMap Demo](#openstreetmap-demo)
* [AirSim Demo](#airsim-demo)


## Installation <a name="installation"/>

The required python version is Python 3.6+.

1. Clone the repository and create and virtual environment with the following lines:
   ```
   git clone git@github.com:h2r/sloop.git
   cd sloop;
   virtualenv -p $(which python3) venv/sloop
   source venv/sloop/bin/activate
   ```

2. Install the `sloop` package.

    ```
    # Assume you're at the root of the sloop repository
    pip install -e .
    ```
    Note that this will install a number of dependencies, including [pomdp-py](https://h2r.github.io/pomdp-py/html/) version 1.2.4.5. See `setup.py` for the list of packages. The `>=` symbol assumes backwards compatibility of those packages.


## Download Dataset & Models <a name="dataset-and-models"/>
There is one dataset and two models.
* The dataset contains OpenStreetMap data and AMT spatial language descriptions and annotations.
  Download the dataset (3.6MB) from  [here](https://drive.google.com/file/d/1K1SRR3rHcM8Jndjhb-YTB5kqefDNYYbH/view?usp=sharing), and place it under `sloop/datasets` and extract there.

  After extraction your directory structure should look like:
  ```
  / # repository root
    sloop/
        ...
        datasets/
          SL_OSM_Dataset/
            amt/
            frame_of_ref/
            ...
  ```

  Check out [this wiki page](https://drive.google.com/file/d/1XfOUa0xtRstUxJHBdNmk4SLJw970-4vV/view?usp=sharing) for documentation about the dataset.



* The models are the frame of reference prediction models. There is a **front** model (for _front_ and _behind) and a **left** model (for _left_ and _right_).
Download the models (36MB) from [here](https://drive.google.com/file/d/1A7Ak00vljHSUjuyzh6DG6QdjPyBACKsV/view?usp=sharing)
and place it under `sloop/oopomdp/experiments/resources`.

  After extraction your directory structure should look like:
  ```
  / # repository root
    sloop/
        ...
        oopomdp/
            experiments/
                resources/
                    models/
                        iter2_ego-ctx-foref-angle:front:austin
                        ...
  ```


## OpenStreetMap Demo <a name="openstreetmap-demo"/>
You can now start a demo of spatial language object search on an OpenStreetMap.p


## Download and Process Results <a name="results"/>


## OpenStreetMap Demo <a name="openstreetmap-demo"/>


## AirSim Demo <a name="airsim-demo"/>










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
