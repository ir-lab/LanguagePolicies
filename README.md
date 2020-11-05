## Language-Conditioned Imitation Learning for Robot Manipulation Tasks
This repository is the official implementation of [Language-Conditioned Imitation Learning for Robot Manipulation Tasks](https://arxiv.org/abs/2010.12083). 

<div style="text-align:center"><img src="system.png" alt="Model figure" width="800"/></div>

When using this code, please cite the paper:
```
@misc{stepputtis2020languageconditioned,
      title={Language-Conditioned Imitation Learning for Robot Manipulation Tasks}, 
      booktitle = {Advances in Neural Information Processing Systems},
      author={Simon Stepputtis and Joseph Campbell and Mariano Phielipp and Stefan Lee and Chitta Baral and Heni Ben Amor},
      year={2020},
      eprint={2010.12083},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
}
```

## Requirements

To install Python requirements:
```setup
pip install -r requirements.txt
```
Further requirements 
- [CoppeliaSim](https://www.coppeliarobotics.com/downloads) (Only for evaluation)
- [ROS 2 Eloquent](https://index.ros.org/doc/ros2/Installation/Eloquent/) (Only for evaluation)
- [PyRep](https://github.com/stepjam/PyRep) (Only for evaluation)
- [Orocos KDL](https://github.com/orocos/orocos_kinematics_dynamics) (Only for data collection)

## Environment Setup
To run the model, you need to download the __dataset__, __pre-trained model__, and other required files. The required files can be downloaded from [here](https://drive.google.com/uc?id=1hxHmeBEWxhaiIFYW4BKpatz_AFnmqNxt).

The downloaded file should be placed next to the root folder of this repository. The folder _LanguagePolicies_ and the extracted _GDrive_ should reside in the same directory. 

## Docker
If you rather look at this code in a Docker container, we provided a Dockerfile with this repository. To build the container, run the following 
```
docker build -t languagepolicies .
```
After the container is successfully built, start it with the following command (please note that the container takes some time to start up fully) 
```
docker run -p 6081:80 -e RESOLUTION=1280x720 --rm languagepolicies
```
After seeing some terminal output, direct your browser to [localhost:6081](http://localhost:6081). This repository is fully set up in _~/Code_, and you can follow the instructions below to train and/or evaluate the model. In the container, you can find a terminal in the start menu under _System Tools -> LXTerminal_.

Please note that data collection and processing is not supported in the docker container.

## Training
To train the model with default parameters, run the following command in this repository's root directory. 
```training
python main.py
```
The trained model will be located in _Data/Model_, and TensorBoard logs will be in _Data/TBoardLog_. Overall, training will take around 35 hours, depending on your hardware. A GPU is not required, and our model has been trained on a node with two _Intel Xeon CPU E5-2699A v4 @ 2.40GHz_. Please note that the usage of a GPU is not beneficial to our model due to the use of a custom RNN loop.

## Evaluation
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
Our model achieves the following performance:

| Picking         | Pouring        | Sequential         |
|---------------- | -------------- | ------------------ |
|     98%         |      85%       | 84%                |

Below is a demonstration of one of our tasks:
<div style="text-align:center"><img src="demo.gif" alt="Model figure" width="500"/></div>

## Contributing 
If you would like to contribute or have any suggestions, you can contact us at sstepput@asu.edu or open an issue on this GitHub repository.

All contributions welcome! All content in this repository is licensed under the MIT license.