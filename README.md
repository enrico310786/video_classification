# Video classification with PyTorchVideo

In this project I show how to train and evaluate a classifier 
model to perform actions classification on video clip.

You can read more about this project on the following Medium post:
https://medium.com/@enrico.randellini/hands-on-video-classification-with-pytorchvideo-dc9cfcc1eb5f

As a complete use case, I show how to perform data cleaning and augmentation 
on the UCF Sports Action Dataset. You can download the dataset
and read its properties from this [link](https://www.crcv.ucf.edu/data/UCF_Sports_Action.php).
The dataset has the following 10 classes: 
Diving, Golf Swing, Kicking, Lifting, Riding Horse, Running, SkateBoarding, Swing-Bench, Swing-Side, Walking.
The videos are in .avi format, thus we have to transform in the .mp4 format 

Once the dataset is collected, cleaned and augmented, I will fine tune 
and test on this dataset on a pretrained model provided by the library [PyTorchVideo](https://pytorchvideo.org/).


## Preprocess

When you download and unzip the UCF Sports Action Dataset, you will note that the Golf Swing class is divided 
into the three Golf-Swing-Back, Golf-Swing-Front and Golf-Swing-Side classes and 
the class Kicking is divided into the two Kicking-Front and Kicking-Side classes. Thus we have to group the
subclasses in a single one.

Furthermore the dataset has the following structure

    videos:
        class_1:
            001:
                video_1
            002:
                video_2
            003:
                video_3
            ...
        class_2:
            001:
                video_1
            002:
                video_2
            003:
                video_3
            ...
        class_3:
            001:
                video_1
            002:
                video_2
            003:
                video_3
            ...
        ...

The final dataset, however, must have the following structure 

    videos:
        class_1:
            video_1
            video_2
            video_3
            ...
        class_2:
            video_1
            video_2
            video_3
            ...
        class_3:
            video_1
            video_2
            video_3
            ...
        ...

Thv Video preprocessing consists of the following steps:

1) clean and create the directories where are generated the final datasets
2) transform the videos from the .avi to the .mp4 format and generate the correct structure for the dataset
3) remove corrupted videos shorter than 1 second and split the dataset into train, validation and test subsets
4) perform video data augmentation by Albumentations. 
   You can set the final number of videos for each category and if you want to apply data augmentation 
   only to the train set or to all the splitted subset
5) create the csv files for train, validation and test dataset with the path to the videos and the correct label.
    These files will be used to create the dataloaders.

The preprocess is runned by the script

```bash
python preprocess_dataset_ucf_action_sport.py
```
Note that during the execution will be generated the dictionary class2label, namely the map from the name of the
category to the corresponding label. The order of the keys of the dictionary, that are the classes of the dataset, has to be respected
at the inference time to obtain the correct classes.

## Train and test

If your device has a GPU, you can run the training and test script
```bash
python run_train_test.py
```
This only requires the configuration file containing the values of the hyperparameters 
of the model and the paths to the datasets.

If you want to make the train procedure on AWS Batch platform, you can use the script
```bash
python run_train_test_aws.py
```
In addition to the configuration file path, you must also enter the name of the S3 bucket 
and the bucket directory where are contained the datasets. 
The download and upload of the data and checkpoints saved during the training is managed 
by the boto3 client using the multidown and multiup functions.
In this case you need to create the docker image of the project, inserting inside also the
credential file with your aws_access_key_id and aws_secret_access_key.

To create the docker image you can use the provided Dockerfile

In the configuration file you can set

1) the path to the dataset and the csv files for train, validation and test. Note that these files
   have to stay inside the dataset directory
2) num_classes: the number of classes to be classified
3) num_epoch: the number of epochs of the training phase
4) batch_size: the size of the batch used by the dataloaders. 
   Depending on the RAM memory of your device, if this value is too high you can obtain CUDA out of memory errors
5) num_frames: the number of evenly sampled frames and used to execute the train and the inference steps
6) learning_rate: the initial learning rate
7) scheduler_step_size: the number of epochs that must pass to change the value of the learning rate
8) scheduler_gamma: the value by which the current learning rate is multiplied to obtain the next learning rate 
9) n_nodes: the number of nodes of the hidden layer between the base model and the output layer   
10) freeze_layers: 1 if the weights of the base model have to be freezed. 0 otherwise
11) epoch_start_unfreeze: the epoch from which you want to unlock the base model weights.
    if you comment out this line the base model weights will never be trained
12) block_start_unfreeze: the base model block from which you intend to unlock the weights
13) do_train: 1 if you want to train the model. 0 otherwise
14) do_test: 1 if you want to test the model. 0 otherwise
15) path_torchvideo_model: the path to the PyTorchVideo repo where download the pretrained model
16) name_torchvideo_model: the name of the pretrained model
17) crop_video: if you want to apply CenterCropVideo with a crop_size=256 to the video during the train and inferece steps

## Inference

One trained your model, with the script 
```bash
python run_train_test_aws.py
```
you can apply inference to some video clips and check the result

## Trained model

You can download my trained model from this [link](https://drive.google.com/file/d/1XtXNCWxLVv7NanNGPCHBqdtiRR-OzGQt/view?usp=share_link)

## Environment

I use Python 3.7.9

To run the scripts in your device create an environment using the file 'requirements.txt'

To run the script on AWS use the file 'requirements_docker.txt' as expressed in the Dockerfile
