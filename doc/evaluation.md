# Training and Evaluation

## Training
Training can be run by running the following command.
```
python main.py
```
This will run the training with our default parameters on the full dataset that you downloaded as part of the [data archive](https://drive.google.com/uc?id=1hxHmeBEWxhaiIFYW4BKpatz_AFnmqNxt). The trained model will be located in _Data/Model_, and TensorBoard logs will be in _Data/TBoardLog_. 


### Custom Training
Should you have collected data following the [Data Collection](data_collection.md) tutorial, you can use your dataset by changed the parameters in _main.py_ according to your needs. The respective parameters can be found in line 18 to 35.

### Hyper-Parameters
The following hyperparameters are used in our default configuration and represent the used learning rate as well as the weights used for the auxiliary losses. 
Furthermore, the ranges for each value indicate the values that have been explored during the hyper-parameter search.
A reasonable range for each parameter was determined before the automated parameter search. 
The automated search for suitable parameters was conducted by _sklearn.model\_selection.RandomizedSearchCV_


| Lerning Rate                | Attention Weight           | Weight difference      | Trajectory Reconstruction     | Phase Progression     | Phase Estimation     |
| --------------------------- | -------------------------- | ---------------------  | ----------------------------- | --------------------- | -------------------- |
| **0.0001** [0.01 - 0.00005] | **1.0** [fixed reference]  | **50.0** [0.1 - 100.0] | **5.0** [0.1 - 30]            | **14.0** [0.1 - 30]   | **1.0** [0.1 - 30]   |

### Pre-Trained Models
Our archived material contains a pre-trained model that you can use to reproduce some of the results. 
It is already part of the docker container or can be downloaded as part of the GDrive archive.
Furthermore, we also include a fine-tuned version of Faster RCNN used in our model to determine object positions and classes. 
Pre-training of Faster RCNN has been done on a single NVIDIA Quadro P6000 GPU over the 40000 samples of the training data in approximately 15 minutes. 
The two external modules that we are using are Faster RCNN and GloVe. 
We are using the following versions:

* **Faster RCNN** Based on ResNet 101 and trained on the COCO dataset
* **GloVe** Trained on Wikipedia 2014 and Gigaword 5 (6B tokens, 400k vocab, 50 dimensions)

## Evaluation
Our model can be evaluated in multiple different ways. The default way is to run the evaluation on the 100 test-data that are provided alongside this repository. However, you can also run in manual mode, allowing you to generate random environments and input your own sentences. 

### Running the Service
In order to run the evaluation, a network service is required that provides the neural network. This service can be started by running. Please have a look at the parameters in line 26 to 37 before running the service.
```
python service.py
```

### Running Automatic Evaluation
Before running the evaluation, please make sure the parameters in line 26 to 39 in _val\_model\_vrep.py_ are set correctly. To run the automatic evaluation, it should be set to RUN_ON_TEST_DATA=True. This will cause VRep to evaluate the model on the 100 test data that we used to generate Table 1 in the paper. 
```
python val_model_vrep.py
```
The results of the evaluation run will be shown at the end of all interactions. After the simulation of the test data has been completed, a _val\_results.json_ will be created in the root directory, containing all the data needed to generate one entry in our paper's results table. You can run 
```
python viz_val_vrep.py
```
 in order to generate one line of Latex code that represents the results for the tested model setting:

```
0.98 & 0.85 & 0.84 & 0.94 & 0.94 & 0.88 & 0.05 & 4.85 & 0.83 & 0.83 & 0.85 & 1.00 & 0.88 & 1.00 & 0.70 & 0.89
```

In order to change the results file, please have a look at the configuration at the top of _viz\_val\_vrep.py_, line 10 to 13. 

Our model's evaluation on all the test scenarios will take 2-3 hours, depending on your hardware. The number of tested demonstrations can be changed in file _val\_model\_vrep.py_ and is set to 100 as default. For reference, we included the test result of our model in _ours\_full\_cl.json_. 

The full results can be seen on our [Additional Results](detailed_results.md) page.

### Running manual evaluation
Before running the evaluation, please make sure the parameters in line 26 to 39 in _val\_model\_vrep.py_ are set correctly. Particularly, RUN_ON_TEST_DATA needs to be set to False. This will allow you to interact with the system directly through your terminal. Please let VRep fully start up until you see the prompt in the terminal you used to start (please note, the service needs to be running)
```
python val_model_vrep.py
```
You can generate random environments by typing "g" and pressing enter:
```
> g
```
To reset the environment, type "r" and hit enter:
```
> r
```
To run the task, start with a "t" and type your command (as an example _pick up the red cup_) afterward before hitting enter:
```
> t pick up the red cup
```
Finally, to end the simulator, type "q" and hit enter:
```
> q
```
