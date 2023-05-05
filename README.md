# Language-Conditioned Imitation Learning for Robot Manipulation Tasks
This repository is the official implementation of [Language-Conditioned Imitation Learning for Robot Manipulation Tasks](https://arxiv.org/abs/2010.12083), which has been accepted to NeurIPS 2020 as spotlight presentation.

<div style="text-align:center"><img src="doc/system.png" alt="Model figure" width="80%"/></div>

When using this code and/or model, we would apprechiate the following citation:
```
@inproceedings{NEURIPS2020_9909794d,
 author = {Stepputtis, Simon and Campbell, Joseph and Phielipp, Mariano and Lee, Stefan and Baral, Chitta and Ben Amor, Heni},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {H. Larochelle and M. Ranzato and R. Hadsell and M.F. Balcan and H. Lin},
 pages = {13139--13150},
 publisher = {Curran Associates, Inc.},
 title = {Language-Conditioned Imitation Learning for Robot Manipulation Tasks},
 url = {https://proceedings.neurips.cc/paper_files/paper/2020/file/9909794d52985cbc5d95c26e31125d1a-Paper.pdf},
 volume = {33},
 year = {2020}
}

```

## Inddex
1. [Environment Setup](#environment-setup)
      - [Local Setup](#local-setup)
      - [Docker](#docker)
2. [Quick Start](#quick-start)
      - [Details: Data Collection](doc/data_collection.md)
      - [Details: Training and Evaluation](doc/evaluation.md)
3. [Results](#results)
      - [Additional Results](doc/detailed_results.md)
      - [Various Demonstrations](doc/demonstrations.md)
4. [Changelog](#changelog)

## Environment Setup
### Local Setup
Our code is tested on __Ubuntu 22.04__ with __Python 3.10__ (Note, Ubuntu 20.04 requries python 3.8). At this time, running our code on MacOS or Windows is not supported, but may work. In order to set up our code create a new conda environment as follows:

```
sudo apt install libcurl4-openssl-dev libssl-dev libeigen3-dev python3-dev mesa-utils libgl1-mesa-glx
```

To install Python requirements (this is for Ubuntu 22.04 and Python 3.10):
```setup
conda env create -f environment.yml 
```
This will set up a basic environment named "lp". The reason for the python version is that if you compile KDL from scratch, it uses the system python, which is 3.10 for Ubuntu 22.04. While not provided, you should be able to run this code on other Linux versions. If you are not using the setup described above, you may need to re-compile the protobuf files, which can be done via the _compile.sh_ script in _utils/proto_.
For the additional setup, please active that environment
```
conda activate lp
```

Further modules are needed and need to be manually installed:
- [CoppeliaSim](https://www.coppeliarobotics.com/downloads): Downloading and installing the player version will be sufficient, as long as you do not want to change the simulation environment itself. Our code was tested with version 4.1 (On Ubuntu 22.04, download the 20.04 version which seems to be working).

After downloading and extracting CoppeliaSim, you will need the following environment variables set (please replace the path accordingly)
```
export COPPELIASIM_ROOT=/<path>/<to>/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

Then, install the remaining dependancies:
- [PyRep](https://github.com/stepjam/PyRep): Please clone their repository and check out commit _96c0b034ee21ab5e6ba0942c4d57993a8670379a_. Then, the package can be installed with: _pip install ._ (note the "." at the end). You can test if PyRep is set up correctly with a small script found in _utils/test_pyrep.py_

- [Orocos KDL](https://github.com/orocos/orocos_kinematics_dynamics): The python-wrapper has to match the solver version installed on your system. We strongly suggest to install both components from the git repository. Our code was tested with commit _86c7893234aeccec3b9bf24cf20de9380d64bdf3_. Please follow their installation instructions for _orocos_kdl_ and _python_orocos_kdl_. However, after running _make_ in the python package, you should have created a file called _PyKDL.so_. Copy this file to your python interpreter's _site-packages_. You can check if KDL is ready to be used by running the code in _utils/test_KDL.py_.


To run the model and simulation, you need to download the __dataset__, __pre-trained model__, and other required files. The required files can be downloaded from [here](https://drive.google.com/file/d/1fE_Tv44Vl40_KNeu9oI-3VllRJVXyNVZ/view?usp=share_link). The downloaded file contains a pre-trained model, the processed training dataset (and other supporting files), and the test-data used for evaluation.
The downloaded file should be placed next to the root folder of this repository. The folder _LanguagePolicies_ and the extracted _GDrive_ should reside in the same directory. 
Additionally we provide our full raw data as an optinoal download [here](https://drive.google.com/file/d/1ssZUdL3PIrppug5kRhwQhP-InCA1y9nY/view?usp=share_link) (Note: 6GB download and ~100GB extracted)

## Quick Start
A detailed description of the training and evaluation process can be found on our [Details: Training and Evaluation](doc/evaluation.md) page. If you are interested in collecting data, please refer to our [Details: Data Collection](doc/data_collection.md) page.

### Training
To train the model with default parameters, run the following command in this repository's root directory. 
```training
python main.py
```
The trained model will be located in _Data/Model_, and TensorBoard logs will be in _Data/TBoardLog_. Overall, training will take around 35 hours, depending on your hardware. A GPU is not required, and our model has been trained on a node with two _Intel Xeon CPU E5-2699A v4 @ 2.40GHz_. Please note that the usage of a GPU is not beneficial to our model due to the use of a custom RNN loop.


### Evaluation
Our model can be live-evaluated in CoppeliaSim. To run the evaluation, ROS2 is required. Please start by building the _ros2_ workspace and source it. First, the pre-trained model will be loaded from the _GDrive_ directory and provided as a service with 
```
python service.py
```
After the service has been started, the model can be evaluated in the simulator with 
```
python val_model_vrep.py
```
This will create a file _val\_result.json_ after ten evaluation runs (Results in our paper are from 100 runs. This value can be changed). Results can be printed in the terminal by running. 
```
python viz_val_vrep.py
```

## Results
We summarize the results of testing our model on a set of 100 unseen, new environments. Our model's overall task success describes the percentage of cases in which the cup was first lifted, and then successfully poured into the correct bowl. This sequence of steps was successfully executed in 84% of the new environments. Picking alone achieves a 98% success rate while pouring results in 85%. The _Detection_ rate indicates the success rate of the semantic model, attempting to identify the correct objects. _Content-In-Bowl_ outlines the percentage of material that was delivered to the correct bowl during the pouring action. Finally, we report the mean-absolute-error of the robot's joint configuration. These results indicate that the model appropriately generalizes the trained behavior to changes in object position, verbal command, or perceptual input. In additon, we also compared the models performance to a simple RNN approach and a recent state-of-the-art baseline ("_Pay attention!-robustifying a deep visuomotor policy through task-focused visual attention_" Abolghasemi et. al.):

| Model              | Picking         | Pouring        | Sequential         | Detection | Content-In-Bowl | MAE (Joints, Radiant)  |
| ------------------ | :-------------: | :------------: | :----------------: | :-------: | :-------------: | :--------: |
| __Simple RNN__     |     58%         |      0%        | 0%                 | 52%       | 7%              | 0.30&#176; |
| __PayAttention!__  |     23%         |      8%        | 0%                 | 66%       | 41%             | 0.13&#176; |
| __Ours__           |     98%         |      85%       | 84%                | 94%       | 94%             | 0.05&#176; |

Further results can be found in our [Additional Results](doc/detailed_results.md) page.

An execution of our model in a specific environment is shown below. First, the languaage command _Rais the green cup_ and an image of the current environment is given to the model. This allows the robot to identify the target object in the current environment, as well as and desired action. After the cup has been picked up, a second comand _Fill all of it into the small red bowl_ is issued and processed in the same environment. In addition to identify the target bowl and action (the _what_ and _where_), the robot also identifies a quantity modifier, used to describe _how_ the robot should execute the described task. In this case, all of the cup's content is filled into the target bowl. 

<div style="float:left">
      <img src="doc/model.gif" alt="" width="49%"/>
      <img src="doc/demo.gif" alt="" width="49%"/>
</div>

More examples can be found in the [Additional Examples](doc/demonstrations.md)

## Contributing 
If you would like to contribute or have any suggestions, feel free to open an issue on this GitHub repository or contact the first author of this work!

All contributions welcome! All content in this repository is licensed under the MIT license.

## Changelog

The following additions were made:
- __May 2023__
  - Updated the code to work with recent versions of various depending libraries
      - Updated KDL version
      - Updated PyRep version
      - Updated to Python 3.10
      - Updated to Ubuntu 22.04
      - Updated scene file to work with newer versions of CoppeliaSim
  - Added a link to the full dataset used for training
  - Removed the Docker version of this repo as it is very outdated
  - Removed the dependency on ROS and replaced it with gRPC as it is much more lightweight
- __November 2020__
  - Initial releas