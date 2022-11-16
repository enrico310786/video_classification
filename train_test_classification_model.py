import torch
import random
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from data import create_loaders
from logger import Logger
from model import VideoClassificationModel
from upload_s3 import multiup
from utils import find_last_checkpoint_file, plot_learning_curves
import torch.nn as nn
from sklearn import metrics
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

post_act = torch.nn.Softmax(dim=1)


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def train_batch(inputs, labels, model, optimizer, criterion):
    model.train()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()


@torch.no_grad()
def accuracy(inputs, labels, model):
    model.eval()
    outputs = model(inputs)
    # Get the predicted classes
    preds = post_act(outputs)
    #print('preds: ', preds)
    _, pred_classes = torch.max(preds, 1)
    #print('pred_classes: ', pred_classes)
    #print('labels:', labels)
    is_correct = pred_classes == labels
    #print('is_correct: ', is_correct)
    return is_correct.cpu().numpy().tolist()


@torch.no_grad()
def val_loss(inputs, labels, model, criterion):
    model.eval()
    outputs = model(inputs)
    val_loss = criterion(outputs, labels)
    return val_loss.item()


def train_model(device,
                model,
                criterion,
                optimizer,
                lr_scheduler,
                classification_dataloader_train,
                classification_dataloader_val,
                best_epoch,
                num_epoch,
                best_val_epoch_accuracy,
                checkpoint_dir,
                saving_dir_experiments,
                logger,
                epoch_start_unfreeze=None,
                block_start_unfreeze=None,
                aws_bucket=None,
                aws_directory=None):

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    print("Start training")
    freezed = True
    for epoch in range(best_epoch, num_epoch):
        logger.log(f'Epoch {epoch}/{num_epoch - 1}')

        if epoch_start_unfreeze is not None and epoch >= epoch_start_unfreeze and freezed:
            print("****************************************")
            print("unfreeze the base model weights")

            if block_start_unfreeze is not None:
                print("unfreeze the layers greater and equal to unfreezing_block: ", block_start_unfreeze)
                #in this case unfreeze only the layers greater and equal the unfreezing_block layer
                for name, param in model.named_parameters():
                    if int(name.split(".")[2]) >= block_start_unfreeze:
                        param.requires_grad = True
            else:
                # in this case unfreeze all the layers of the model
                print("unfreeze all the layer of the model")
                for name, param in model.named_parameters():
                    param.requires_grad = True
            freezed = False

            for name, param in model.named_parameters():
                print("Layer name: {} - requires_grad: {}".format(name, param.requires_grad))
            print("****************************************")

        # define empty lists for the values of the loss and the accuracy of train and validation obtained in the batch of the current epoch
        # then at the end I take the average and I get the final values of the whole era
        train_epoch_losses, train_epoch_accuracies = [], []
        val_epoch_losses, val_epoch_accuracies = [], []

        # iterate on all train batches of the current epoch by executing the train_batch function
        for inputs, labels in tqdm(classification_dataloader_train, desc=f"epoch {str(epoch)} | train"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            batch_loss = train_batch(inputs, labels, model, optimizer, criterion)
            train_epoch_losses.append(batch_loss)
        train_epoch_loss = np.array(train_epoch_losses).mean()

        # iterate on all train batches of the current epoch by calculating their accuracy
        for inputs, labels in tqdm(classification_dataloader_train, desc=f"epoch {str(epoch)} | train"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            is_correct = accuracy(inputs, labels, model)
            train_epoch_accuracies.extend(is_correct)
        train_epoch_accuracy = np.mean(train_epoch_accuracies)

        # iterate on all batches of val of the current epoch by calculating the accuracy and the loss function
        for inputs, labels in tqdm(classification_dataloader_val, desc=f"epoch {str(epoch)} | val"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            val_is_correct = accuracy(inputs, labels, model)
            val_epoch_accuracies.extend(val_is_correct)
            validation_loss = val_loss(inputs, labels, model, criterion)
            val_epoch_losses.append(validation_loss)
        val_epoch_accuracy = np.mean(val_epoch_accuracies)
        val_epoch_loss = np.mean(val_epoch_losses)

        phase = 'train'
        logger.log(f'{phase} LR: {lr_scheduler.get_last_lr()} - Loss: {train_epoch_loss:.4f} - Acc: {train_epoch_accuracy:.4f}')
        phase = 'val'
        logger.log(f'{phase} LR: {lr_scheduler.get_last_lr()} - Loss: {val_epoch_loss:.4f} - Acc: {val_epoch_accuracy:.4f}')
        print("Epoch: {} - LR:{} - Train Loss: {:.4f} - Train Acc: {:.4f} - Val Loss: {:.4f} - Val Acc: {:.4f}".format(int(epoch), lr_scheduler.get_last_lr(), train_epoch_loss, train_epoch_accuracy, val_epoch_loss, val_epoch_accuracy))
        logger.log("-----------")

        train_losses.append(train_epoch_loss)
        train_accuracies.append(train_epoch_accuracy)
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_accuracy)

        print("Plot learning curves")
        plot_learning_curves(epoch - best_epoch + 1, train_losses, val_losses, train_accuracies, val_accuracies, checkpoint_dir)

        if best_val_epoch_accuracy < val_epoch_accuracy:
            print("We have a new best model! Save the model")

            #update best_val_epoch_accuracy
            best_val_epoch_accuracy = val_epoch_accuracy
            save_obj = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'best_eva_accuracy': best_val_epoch_accuracy
            }
            print("Save best checkpoint at: ", os.path.join(checkpoint_dir, 'best.pth'))
            torch.save(save_obj, os.path.join(checkpoint_dir, 'best.pth'),  _use_new_zipfile_serialization=False)
            print("Save checkpoint at: ", os.path.join(checkpoint_dir, 'checkpoint_' + str(epoch) + '.pth'))
            torch.save(save_obj, os.path.join(checkpoint_dir, 'checkpoint_' + str(epoch) + '.pth'),  _use_new_zipfile_serialization=False)

        else:
            print("Save the current model")

            save_obj = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'best_eva_accuracy': best_val_epoch_accuracy
            }
            print("Save checkpoint at: ", os.path.join(checkpoint_dir, 'checkpoint_' + str(epoch) + '.pth'))
            torch.save(save_obj, os.path.join(checkpoint_dir, 'checkpoint_' + str(epoch) + '.pth'),  _use_new_zipfile_serialization=False)

        if aws_bucket is not None and aws_directory is not None:
            print('Upload on S3')
            multiup(aws_bucket, aws_directory, saving_dir_experiments)

        lr_scheduler.step()
        torch.cuda.empty_cache()
        print("---------------------------------------------------------")

    print("End training")
    return


def test_model(device,
               model,
               classification_dataloader,
               path_save,
               class2label,
               type_dataset):

    y_test_true = []
    y_test_predicted = []
    total = 0
    model = model.eval()

    with torch.no_grad():

        # cycle on all train batches of the current epoch by calculating their accuracy
        for inputs, labels in classification_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            # Get the predicted classes
            preds = post_act(outputs)
            # print('preds: ', preds)
            _, pred_classes = torch.max(preds, 1)
            y_test_true.extend(labels.cpu().numpy().tolist())
            y_test_predicted.extend(pred_classes.cpu().numpy().tolist())
            numero_video = len(labels.cpu().numpy().tolist())
            total += numero_video

        # report predictions and true values to numpy array
        print('Number of tested videos: ', total)
        y_test_true = np.array(y_test_true)
        y_test_predicted = np.array(y_test_predicted)
        print('y_test_true.shape: ', y_test_true.shape)
        print('y_test_predicted.shape: ', y_test_predicted.shape)

        print('Accuracy: ', accuracy_score(y_test_true, y_test_predicted))
        print(metrics.classification_report(y_test_true, y_test_predicted))

        ## Plot confusion matrix
        cm = metrics.confusion_matrix(y_test_true, y_test_predicted)

        fig, ax = plt.subplots(figsize=(50, 30))
        sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues,
                    cbar=False)
        ax.set(xlabel="Pred", ylabel="True", xticklabels=class2label.keys(),
               yticklabels=class2label.keys(), title="Confusion matrix")
        plt.yticks(rotation=0)
        fig.savefig(os.path.join(path_save, type_dataset + "_confusion_matrix.png"))

        ## Save report in a txt
        target_names = list(class2label.keys())
        cr = metrics.classification_report(y_test_true, y_test_predicted, target_names=target_names)
        f = open(os.path.join(path_save, type_dataset + "_report.txt"), 'w')
        f.write('Title\n\nClassification Report\n\n{}'.format(cr))
        f.close()


def run_train_test_model(cfg, do_train, do_test, aws_bucket=None, aws_directory=None):

    seed_everything(42)
    checkpoint = None
    best_epoch = 0
    best_val_epoch_accuracy = 0

    dataset_path = cfg['dataset_path']
    path_dataset_train_csv = cfg['path_dataset_train_csv']
    path_dataset_val_csv = cfg['path_dataset_val_csv']
    path_dataset_test_csv = cfg.get("path_dataset_test_csv", None)

    saving_dir_experiments = cfg['saving_dir_experiments']
    saving_dir_model = cfg['saving_dir_model']

    num_epoch = cfg['num_epoch']
    learning_rate = cfg['learning_rate']
    scheduler_step_size = cfg['scheduler_step_size']
    scheduler_gamma = cfg['scheduler_gamma']
    epoch_start_unfreeze = cfg.get("epoch_start_unfreeze", None)
    block_start_unfreeze = cfg.get("block_start_unfreeze", None)

    # 1 - load csv dataset
    path_dataset_train_csv = os.path.join(dataset_path, path_dataset_train_csv)
    df_dataset_train = pd.read_csv(path_dataset_train_csv)
    path_dataset_val_csv = os.path.join(dataset_path, path_dataset_val_csv)
    df_dataset_val = pd.read_csv(path_dataset_val_csv)
    df_dataset_test = None
    if path_dataset_test_csv is not None:
        path_dataset_test_csv = os.path.join(dataset_path, path_dataset_test_csv)
        df_dataset_test = pd.read_csv(path_dataset_test_csv)

    # 2 -  create the directory to save the results and checkpoints
    print("Create the project structure")
    print("saving_dir_experiments: ", saving_dir_experiments)
    saving_dir_model = os.path.join(saving_dir_experiments, saving_dir_model)
    print("saving_dir_model: ", saving_dir_model)
    os.makedirs(saving_dir_experiments, exist_ok=True)
    os.makedirs(saving_dir_model, exist_ok=True)

    # 3 - load log configuration
    logger = Logger(exp_path=saving_dir_model)

    # 4 - create the dataloaders
    classification_dataloader_train, classification_dataloader_val, classification_dataloader_test = create_loaders(df_dataset_train, df_dataset_val, df_dataset_test, cfg)

    # 5 - set the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    # 6 - download the model
    model = VideoClassificationModel(cfg).to(device)
    checkpoint_dir = saving_dir_model

    if do_train:
        # 7 - look if exist a checkpoint
        path_last_checkpoint = find_last_checkpoint_file(checkpoint_dir)
        if path_last_checkpoint is not None:
            print("Carico il best checkpoint più recente al path: ", path_last_checkpoint)
            checkpoint = torch.load(path_last_checkpoint, map_location=torch.device(device))
            model.load_state_dict(checkpoint['model'])
            model = model.to(device)

        #summary of the model
        #summary(model, input_size=(3, 32, 256, 256))

        # 8 - Set the Loss, optimizer and scheduler. ( CrossEntropyLoss wants logits. Perform the softmax internally)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
        if checkpoint is not None:
            print('Load the optimizer from the last checkpoint')
            optimizer.load_state_dict(checkpoint['optimizer'])
            exp_lr_scheduler.load_state_dict(checkpoint["scheduler"])

            print('Latest epoch of the checkpoint: ', checkpoint['epoch'])
            print('Setting the new starting epoch: ', checkpoint['epoch'] + 1)
            best_epoch = checkpoint['epoch'] + 1

            print('Setting best_val_epoch_accuracy from checkpoint: ', checkpoint['best_eva_accuracy'])
            best_val_epoch_accuracy = checkpoint['best_eva_accuracy']

        # 9 - run train model function
        train_model(device=device,
                    model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    lr_scheduler=exp_lr_scheduler,
                    classification_dataloader_train=classification_dataloader_train,
                    classification_dataloader_val=classification_dataloader_val,
                    best_epoch=best_epoch,
                    num_epoch=num_epoch,
                    best_val_epoch_accuracy=best_val_epoch_accuracy,
                    checkpoint_dir=checkpoint_dir,
                    saving_dir_experiments=saving_dir_experiments,
                    logger=logger,
                    epoch_start_unfreeze=epoch_start_unfreeze,
                    block_start_unfreeze=block_start_unfreeze,
                    aws_bucket=aws_bucket,
                    aws_directory=aws_directory)
        print("-------------------------------------------------------------------")
        print("-------------------------------------------------------------------")

    if do_test:

        print("Execute Inference on Train, Val and Test Dataset with best checkpoint")

        path_last_checkpoint = find_last_checkpoint_file(checkpoint_dir=checkpoint_dir, use_best_checkpoint=True)
        if path_last_checkpoint is not None:
            print("Upload the best checkpoint at the path: ", path_last_checkpoint)
            checkpoint = torch.load(path_last_checkpoint, map_location=torch.device(device))
            model.load_state_dict(checkpoint['model'])
            model = model.to(device)

        class2label = {}
        # go through the lines of the dataset
        for index, row in df_dataset_train.iterrows():
            class_name = row["CLASS"]
            label = row["LABEL"]

            if class_name not in class2label:
                class2label[class_name] = label

        print("class2label: ", class2label)

        print("-------------------------------------------------------------------")
        print("-------------------------------------------------------------------")

        # 12 - execute the inferences on the train, val and test set
        print("Inference on train dataset")
        test_model(device=device,
                   model=model,
                   classification_dataloader=classification_dataloader_train,
                   path_save=checkpoint_dir,
                   class2label=class2label,
                   type_dataset="train")

        print("-------------------------------------------------------------------")
        print("-------------------------------------------------------------------")

        print("Inference on val dataset")
        test_model(device=device,
                   model=model,
                   classification_dataloader=classification_dataloader_val,
                   path_save=checkpoint_dir,
                   class2label=class2label,
                   type_dataset="val")

        if classification_dataloader_test is not None:
            print("-------------------------------------------------------------------")
            print("-------------------------------------------------------------------")

            print("Inference on test dataset")
            test_model(device=device,
                       model=model,
                       classification_dataloader=classification_dataloader_test,
                       path_save=checkpoint_dir,
                       class2label=class2label,
                       type_dataset="test")

        if aws_bucket is not None and aws_directory is not None:
            print("Final upload on S3")
            multiup(aws_bucket, aws_directory, saving_dir_experiments)