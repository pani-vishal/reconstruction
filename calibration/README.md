# pgII-calibration :dart:
The project aims to calibrate the camera retrofitted to the gonio-photometer in the [RGLlab](http://rgl.epfl.ch/pages/lab/pgII). This will allow the system to produce captures that can be used in downstream tasks like 3D object reconstruction.

## Project Setup :construction:
The project environment can produced using miniconda and the [environment.yml](environment.yml) file. Instructions to install miniconda can be found [here](https://docs.conda.io/en/latest/miniconda.html).

Once done, 

1. Move to this directory:
```console
foo@bar:~$ cd path/to/this/directory
``` 
2. Create a new environment with the following command:
```console
foo@bar:~$ conda env create --name <env_name> -f environment.yml 
```
3. Activate the environment using the command:
```console 
foo@bar:~$ conda activate <env_name>
```

## Directory Structure :bookmark_tabs:

    .
    ├── analysis.py                 Script to analyze the optimized camera model   
    ├── optimize.py                 Script to optimize the camera model
    ├── process.py                  Script to process the images given the parameters of a camera model
    |
    ├── data                        !!Must be created manually!! Contains the data that the project requires
    |   ├── configs_analysis        !!Must be created manually!! Contains the json configs required to run analysis.py
    |   ├── configs_optimize        !!Must be created manually!! Contains the json configs required to run optimize.py
    |   ├── configs_process         !!Must be created manually!! Contains the json configs required to run process.py
    |   ├── <dataset_1>.pkl         A pickle file of a capture dataset (it's structure is provided in a later section)
    |   └── ...
    |
    ├── helpers                 
    │   ├── beautify_params.py      A hardcoded script to convert the camera model params (np.array) into a readable json
    │   ├── camera_model.py         Script containing functions for the camera model
    │   ├── img_process.py          Script containing functions for image processing
    │   ├── io.py                   Script containing functions for the input/output tasks
    │   ├── json_analysis.py        Script to write JSON config files for the analysis.py script
    │   ├── json_optmize.py         Script to write JSON config files for the optimization.py script
    │   ├── json_process.py         Script to write JSON config files for the process.py script
    │   ├── linear_alg.py           Script containing the linear algebraic functions
    │   └── visualize.py            Script containing functions for all the visualization tasks
    |
    ├── outputs                     !!Must be created manually!! Contains the outputs that the project produces
    |   ├── analysis_stats          !!Must be created manually!! Folder containing the statistics produced by analysis.py                              
    |   ├── params                  !!Must be created manually!! Folder containing the optimized parameters
    |   ├── plots                   !!Must be created manually!! Folder containing all the plots
    |   └── processed               !!Must be created manually!! Folder containing all the processed images
    └── generate_aruco.py           Script to generate a pickle file for the aruco sticker detections.
