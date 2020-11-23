# Data Collection

Before getting started with this tutorial, please complete the installation from the _Environment Setup_.

## Running Data Collection
We provide the code for data collection (please make sure to have installed PyKDL from the _Environment Setup_). Before running data collection, please adjust the parameters in _utils/collect\_data.py_ as desired (Lin 19 to 40). A brief description of the parameters can be found in the python file. 

After adjusting the parameters, data collection can be started by running the following command from within LanguagePolicies:

```
python utils/collect_data.py
```

This code can also be run on a computer without an X-Server. To do so, please install _xvfb_ and run the data collection as follows:
```
xvfb-run -a python utils/collect_data.py
```

After the data collection is finished, the JSON files will be in your specified target directory. The format of these files is explained [below](#data\ format).

## Running Data Processing
While the data collected in the "Data Collection" section can be used for evaluation, further processing is required to convert them into TFRecord files for training. Before running data processing, please adjust the parameters in line 255 to 280 to match your setup. 
```
python utils/data_processing.py
```
Please not that at least one bacht with 16 demonstrations is required for the training and validation data.

## Custom FRCNN
Please note that data processing is already running Faster-RCNN to detect the objects in the scene. This speeds up training time since Faster-RCNN is a frozen model at this point already. During evaluation, a frozen Faster-RCNN is part of our end-to-end architecture. However, it can be run during data processing to speed up training times. To use a custom FRCNN, specify the path to the trained model in line 256.

## Data Format
Due to size limitations, we do not provide the unprocessed dataset (150 GB). Instead, the processed dataset is presented in the two TFRecord files train.tfrecord and validate.tfrecord. The test data are provided as raw data. All data can be found in the GDrive archive that is inside the docker container or can alternatively be downloaded [here](https://drive.google.com/uc?id=1hxHmeBEWxhaiIFYW4BKpatz_AFnmqNxt). Overall, our dataset contains 22,500 complete task demonstrations, composed of the two sub-tasks (grasping and pouring), resulting in 45,000 samples. Of these samples, we use 4,000 for validation and 1,000 for testing, leaving 40,000 for training. The results are reported based on 100 complete tasks that have been randomly selected from the test set (these can be found in the downloaded data above). Each of these 100 tasks consists of two actions, resulting in 200 interactions.

Data is collected in a JSON format, saving all relevant information about the environment, the verbal command, and the task execution itself. Below is an explanation of the file structure

- __amount__: Describing the amount poured. 110 degrees for _some_ and 180 for _all_.
- __target/id__: The id of the target object. See the table below for id descriptions
- __target/type__: Type of the target, either _cup_ or _bowl_
- __trajectory__: The initially generated trajectory that is executed by the robot during data collection. This trajectory has been automatically generated to fulfill the desired task.
- __name__: The name of this demonstration. Overall, there will be two files with the same name. One for the pocking, one for the pouring action.
- __phase__: Either 0 or 1, where 0 indicates the picking and 1 indicates the pouring action. Note that the file names are using 1 for picking and 2 for pouring.
- __image__: An array containing the top-down image of the environment in uint8 format. 
- __ints__: Describes how many and which cups and bowls are in the environment. Index 0 holds the number of bowls, index 1 the number of cups, followed by the bowl and cup ids used. See the table below for a description of the ids.
- __floats__: For each bowl and cup, there are three values. The first two describe the x/y position of the object in the robot coordinate frame; the third value describes its rotation around the z-axis. 
- __state/raw__: Holds the raw robot state recorded during data collection when executing the trajectory given in _trajectroy_. The values are as follows: 
    - 6x robot joint position (j1, j2, j3, j4, j5, j6)
    - 6x robot joint velocity (j1, j2, j3, j4, j5, j6) (Not Used)
    - 3x robot tool-center-point position (x, y, z) (Not Used)
    - 3x robot tool-center-point rotation (x, y, z) (Not Used)
    - 3x robot tool-center-point linear velocity (x, y, z) (Not Used)
    - 3x robot tool-center-point angular velocity (x, y, z) (Not Used)
    - 3x robot tool-center-point target position (x, y, z) (Not Used)
    - 3x robot tool-center-point target rotation (x, y, z) (Not Used)
    - 1x gripper position
    - 1x gripper joint velocity (Not Used)
- __state/dict__: Same as state/raw, but as a parsed dictionary. See state/raw for descriptions
- __voice__: The voice command used for this demonstration

In our experiments, we use the following mapping for bowls and cups:
| ID | Type | Color  | Size  | Shape  |
| -- | :--: | :----: | :---: | :----: |
| 1  | cup  | red    |  n/a  |  n/a   |
| 2  | cup  | green  |  n/a  |  n/a   |
| 3  | cup  | blue   |  n/a  |  n/a   |
| 1  | bowl | yellow | small | round  |
| 2  | bowl | red    | small | round  |
| 3  | bowl | green  | small | round  |
| 4  | bowl | blue   | small | round  |
| 5  | bowl | pink   | small | round  |
| 6  | bowl | yellow | large | round  |
| 7  | bowl | red    | large | round  |
| 8  | bowl | green  | large | round  |
| 9  | bowl | blue   | large | round  |
| 10 | bowl | pink   | large | round  |
| 11 | bowl | yellow | small | square |
| 12 | bowl | red    | small | square |
| 13 | bowl | green  | small | square |
| 14 | bowl | blue   | small | square |
| 15 | bowl | pink   | small | square |
| 16 | bowl | yellow | large | square |
| 17 | bowl | red    | large | square |
| 18 | bowl | green  | large | square |
| 19 | bowl | blue   | large | square |
| 20 | bowl | pink   | large | square |

