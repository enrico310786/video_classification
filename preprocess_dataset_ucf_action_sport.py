"""
The script navigates the dataset directory taking the .avi files
Convert .avi files to .mp4 format
Save the .mp4 files in a new directory with the same tree but containing only the .mp4 files
"""

import os
import argparse
from pytorchvideo.data.encoded_video import EncodedVideo
import random
import shutil
import cv2
import albumentations as A
import pandas as pd
from utils import create_augmented_video, load_video, augment_frames

remapping_actions_dict = {"Golf-Swing-Back": "Golf-Swing",
                          "Golf-Swing-Front": "Golf-Swing",
                          "Golf-Swing-Side": "Golf-Swing",
                          "Kicking-Front": "Kicking",
                          "Kicking-Side": "Kicking"}

PERC_TRAIN = 0.5
PERC_TEST = 0.4
NUMBER_VIDEO_TRAIN = 100
NUMBER_VIDEO_VAL = 5
#NUMBER_VIDEO_TEST = 14

def make_data_augmentation(path_original_dataset, dir_path_augmented, type, final_number=0, apply_augmentation=True):

    path_original_dataset = os.path.join(path_original_dataset, type)
    dir_path_augmented = os.path.join(dir_path_augmented, type)

    for subdir, dirs, files in os.walk(path_original_dataset):
        for dir in dirs:
            path_subdir_original_dataset = os.path.join(path_original_dataset, dir)
            CHECK_FOLDER = os.path.isdir(path_subdir_original_dataset)
            if CHECK_FOLDER:
                print("CLASSE: ", dir)

                number_files = len(os.listdir(path_subdir_original_dataset))
                print("Number_files on directory '{}': {}".format(path_subdir_original_dataset, number_files))

                # create the new directory where save the original file and its augmentation
                path_directory_save = os.path.join(dir_path_augmented, dir)
                CHECK_FOLDER = os.path.isdir(path_directory_save)
                if not CHECK_FOLDER:
                    os.makedirs(path_directory_save)

                if not apply_augmentation:
                    # iterate over the list of path
                    for file_name in os.listdir(path_subdir_original_dataset):
                        path_video = os.path.join(path_subdir_original_dataset, file_name)
                        new_file_name = str(file_name.split(".")[0]) + "_0.mp4"
                        new_file_path = os.path.join(path_directory_save, new_file_name)
                        print("Copio il file originale '{}' in '{}' ".format(path_video, new_file_path))
                        shutil.copyfile(path_video, new_file_path)
                else:
                    # determine the number of times I have to apply the transformation on a single video
                    n_applications = round((final_number - number_files) / number_files)
                    print('n_applications: ', n_applications)

                    # iterate over the list of path
                    for file_name in os.listdir(path_subdir_original_dataset):
                        path_video = os.path.join(path_subdir_original_dataset, file_name)
                        # 1: load the video and divide it into frames
                        frame_list, _, fps, _ = load_video(path_video)

                        for i in range(n_applications):
                            # 2: augment each frames
                            augmented_frame_list = augment_frames(frame_list)

                            # 3: generate the new video with the augmented framse
                            name_augmented_video = str(file_name.split(".")[0]) + "_" + str(i+1) + ".mp4"
                            path_augmented_video = os.path.join(path_directory_save, name_augmented_video)
                            create_augmented_video(augmented_frame_list, path_augmented_video, fps)

                        new_file_name = str(file_name.split(".")[0]) + "_0.mp4"
                        new_file_path = os.path.join(path_directory_save, new_file_name)
                        print("Copio il file originale '{}' in '{}' ".format(path_video, new_file_path))
                        shutil.copyfile(path_video, new_file_path)

                    print("Number of final videos for class {} : {}".format(dir, int(len(os.listdir(path_directory_save)))))
                    print("-----------------------------------------------")


def convert_avi_to_mp4(avi_file_path, output_name):
    os.popen("ffmpeg -i '{input}' -ac 2 -b:v 2000k -c:a aac -c:v libx264 -b:a 160k -vprofile high -bf 0 -strict experimental -f mp4 '{output}'".format(input=avi_file_path, output=output_name))
    return True


def create_mp4(dir_dataset, dir_dataset_new):
    counter = 0
    for subdir, dirs, files in os.walk(dir_dataset):
        for dir in dirs:
            path_subdir = os.path.join(dir_dataset, dir)
            CHECK_FOLDER = os.path.isdir(path_subdir)
            if CHECK_FOLDER:
                class_name = path_subdir.split("/")[-1]
                print("class_name: ", class_name)

                if class_name in remapping_actions_dict:
                    class_name = remapping_actions_dict[class_name]
                    print("grouped class_name: ", class_name)

                new_dir_path = os.path.join(dir_dataset_new, class_name)
                CHECK_FOLDER = os.path.isdir(new_dir_path)
                if not CHECK_FOLDER:
                    print("Creo la directory : ", new_dir_path)
                    os.makedirs(new_dir_path)

                for sub_subdir, sub_dirs, sub_files in os.walk(path_subdir):
                    for name_file in sub_files:
                        if name_file.endswith('.avi'):
                            name_file_no_ext = name_file.split(".")[0]
                            path_file_avi = os.path.join(sub_subdir, name_file)
                            path_file_mp4 = os.path.join(new_dir_path, name_file_no_ext + ".mp4")
                            print("Take the .avi file: {} - Generate the .mp4 file {}: ".format(path_file_avi, path_file_mp4))
                            convert_avi_to_mp4(path_file_avi, path_file_mp4)
                            counter += 1

                print("------------------------------------")

    print("Total .mp4 video generated: ", counter)


def analyze_video_duration(video_path):
    video = EncodedVideo.from_path(video_path)
    clip_duration = int(video.duration)
    is_zero_len = False

    if clip_duration == 0:
        print("Video with zero duration - path: {}".format(video_path))
        is_zero_len = True
    return is_zero_len


def make_train_test_val_division(path_original_dataset, dir_path_ttv):

    print("Partition into TRAIN - TEST - VAL")
    for subdir, dirs, files in os.walk(path_original_dataset):
        for dir in dirs:
            path_subdir = os.path.join(path_original_dataset, dir)
            print("CLASS: {}".format(dir))

            # collect the path for mp4 files with duration grater then 1 sec
            list_path_mp4_files = []
            for file in os.listdir(path_subdir):
                path_file = os.path.join(path_subdir, file)
                if not analyze_video_duration(path_file):
                    list_path_mp4_files.append(path_file)

            number_files = len(list_path_mp4_files)
            index_list = [i for i in range(number_files)]
            print("Number_files on directory '{}' with duration grater then 1 sec: {}".format(path_subdir, number_files))

            number_file_train = round(number_files*PERC_TRAIN)
            number_file_test = round(number_files*PERC_TEST)

            sum_train_test = number_file_train + number_file_test
            #if the sum of the train and test files equals the total number,
            # I decrease the number of train files by one, so I have space for a validation file
            if sum_train_test == number_files:
                if sum_train_test != 2:
                    number_file_train = number_file_train - 1

            if sum_train_test != 2:
                train_index_list = random.sample(index_list, k=number_file_train)
                index_list = list(set(index_list) - set(train_index_list))
                test_index_list = random.sample(index_list, k=number_file_test)
                val_index_list = list(set(index_list) - set(test_index_list))
            else:
                train_index_list = [0]
                test_index_list = [1]
                val_index_list = [1]

            #TRAIN assignment and copy to train directory
            for idx in train_index_list:
                path_file = list_path_mp4_files[idx]
                filename = path_file.split("/")[-1]
                dst_dir = os.path.join(dir_path_ttv, "train", dir, filename)

                CHECK_FOLDER = os.path.isdir(os.path.join(dir_path_ttv, "train", dir))
                if not CHECK_FOLDER:
                    os.makedirs(os.path.join(dir_path_ttv, "train", dir))

                #print("Copia immagine da '{}' a {} ".format(src_dir, dst_dir))
                shutil.copy2(path_file, dst_dir)
            print("Number of files into dir '{}': {}".format(os.path.join(dir_path_ttv, "train", dir), len(os.listdir(os.path.join(dir_path_ttv, "train", dir)))))

            #TEST assignment and copy to test directory
            for idx in test_index_list:
                path_file = list_path_mp4_files[idx]
                filename = path_file.split("/")[-1]
                dst_dir = os.path.join(dir_path_ttv, "test", dir, filename)

                CHECK_FOLDER = os.path.isdir(os.path.join(dir_path_ttv, "test", dir))
                if not CHECK_FOLDER:
                    os.makedirs(os.path.join(dir_path_ttv, "test", dir))

                #print("Copia immagine da '{}' a {} ".format(src_dir, dst_dir))
                shutil.copy2(path_file, dst_dir)
            print("Number of files into dir '{}': {}".format(os.path.join(dir_path_ttv, "test", dir),len(os.listdir(os.path.join(dir_path_ttv, "test", dir)))))

            #VAL assignment and copy to val directory
            for idx in val_index_list:
                path_file = list_path_mp4_files[idx]
                filename = path_file.split("/")[-1]
                dst_dir = os.path.join(dir_path_ttv, "val", dir, filename)

                CHECK_FOLDER = os.path.isdir(os.path.join(dir_path_ttv, "val", dir))
                if not CHECK_FOLDER:
                    os.makedirs(os.path.join(dir_path_ttv, "val", dir))

                #print("Copia immagine da '{}' a {} ".format(src_dir, dst_dir))
                shutil.copy2(path_file, dst_dir)
            print("Number of files into dir '{}': {}".format(os.path.join(dir_path_ttv, "val", dir), len(os.listdir(os.path.join(dir_path_ttv, "val", dir)))))

            print('----------------------------------------------------------------')


def create_dict_class2label(dir_path_augmented):
    class2label = dict()
    label = 0
    dir_path = os.path.join(dir_path_augmented, "train")
    for subdir, dirs, files in os.walk(dir_path):
        for dir in dirs:
            path_subdir = os.path.join(dir_path, dir)
            CHECK_FOLDER = os.path.isdir(path_subdir)
            if CHECK_FOLDER:
                class_name = path_subdir.split("/")[-1]
                print("class_name: ", class_name)

                if class_name not in class2label:
                    class2label[class_name] = label
                    label += 1
    return class2label


def create_csv(dir_path_augmented, type, class2label):

    df = pd.DataFrame(columns=['CLASS', 'LABEL', 'PATH', 'NUM_FRAMES', 'NUM_SEC', 'FPS'])
    dir_path = os.path.join(dir_path_augmented, type)

    for subdir, dirs, files in os.walk(dir_path):
        for dir in dirs:
            path_subdir = os.path.join(dir_path, dir)
            CHECK_FOLDER = os.path.isdir(path_subdir)
            if CHECK_FOLDER:
                class_name = path_subdir.split("/")[-1]
                for name_file in os.listdir(path_subdir):
                    if name_file.endswith('.mp4'):
                        video_path = os.path.join(dir_path, dir, name_file)
                        _, num_frames, fps, video_duration = load_video(video_path)
                        relative_path = os.path.join(type, class_name, name_file)
                        df = df.append({'CLASS': class_name,
                                        'LABEL': class2label[class_name],
                                        'PATH': relative_path,
                                        'NUM_FRAMES': num_frames,
                                        'NUM_SEC': video_duration,
                                        'FPS': fps}, ignore_index=True)
    return df


def delete_create_dir(path_dir):
    CHECK_FOLDER = os.path.isdir(path_dir)
    if CHECK_FOLDER:
        print("Delete the directory '{}'".format(path_dir))
        try:
            shutil.rmtree(dir_dataset_grouped_mp4)
        except OSError as e:
            print("Error: {}".format(e.strerror))
            raise e

        print("Create the directory '{}'".format(path_dir))
        os.makedirs(path_dir)
    else:
        print("Create the directory '{}'".format(path_dir))
        os.makedirs(path_dir)
########################################
########################################


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_dataset',
                        type=str,
                        default="/mnt/disco_esterno/ucf_sports_actions/ucf action",
                        help='Directory of the starting ucf sport action dataset with .avi files')
    parser.add_argument('--dir_dataset_grouped_mp4',
                        type=str,
                        default="/mnt/disco_esterno/ucf_sports_actions/ucf_action_grouped_mp4",
                        help='Directory of the ucf sport action dataset with .mp4 files with grouped action')
    parser.add_argument('--dir_dataset_grouped_ttv_mp4',
                        type=str,
                        default="/mnt/disco_esterno/ucf_sports_actions/ucf_action_grouped_ttv_mp4",
                        help='Directory of the ucf sport action dataset with .mp4 files with grouped action separated in train, test and val subdir')
    parser.add_argument('--dir_dataset_grouped_augmented_ttv_mp4',
                        type=str,
                        default="/mnt/disco_esterno/ucf_sports_actions/ucf_action_grouped_augmented_ttv_mp4",
                        help='Directory of the ucf sport action dataset with .mp4 files with grouped action separated in train, test and val subdir and train augmented')
    parser.add_argument('--path_dataset_train_csv',
                        type=str,
                        default="dataset/ucf_action_grouped_augmented/df_train.csv",
                        help='Path where is saved the csv of the train dataset')
    parser.add_argument('--path_dataset_val_csv',
                        type=str,
                        default="dataset/ucf_action_grouped_augmented/df_val.csv",
                        help='Path where is saved the csv of the val dataset')
    parser.add_argument('--path_dataset_test_csv',
                        type=str,
                        default="dataset/ucf_action_grouped_augmented/df_test.csv",
                        help='Path where is saved the csv of the test dataset')

    opt = parser.parse_args()

    dir_dataset = opt.dir_dataset
    dir_dataset_grouped_mp4 = opt.dir_dataset_grouped_mp4
    dir_dataset_grouped_ttv_mp4 = opt.dir_dataset_grouped_ttv_mp4
    dir_dataset_grouped_augmented_ttv_mp4 = opt.dir_dataset_grouped_augmented_ttv_mp4
    path_dataset_train_csv = opt.path_dataset_train_csv
    path_dataset_val_csv = opt.path_dataset_val_csv
    path_dataset_test_csv = opt.path_dataset_test_csv

    # 0 -check directories
    # Check if exist the directory 'dir_dataset_grouped_mp4'. If exist I delete it and recreate it from new
    delete_create_dir(dir_dataset_grouped_mp4)
    # Check if exist the directory 'dir_dataset_grouped_ttv_mp4'. If exist I delete it and recreate it from new
    delete_create_dir(dir_dataset_grouped_ttv_mp4)
    # Check if exist the directory 'dir_dataset_grouped_augmented_ttv_mp4'. If exist I delete it and recreate it from new
    delete_create_dir(dir_dataset_grouped_augmented_ttv_mp4)

    # 1) Transform .a files in .mp4. Collect only mp4 files and group golf action and Kicking action
    create_mp4(dir_dataset, dir_dataset_grouped_mp4)

    print("****************************************************")
    print("****************************************************")

    # 2) Split the dataset into train, test and val subdir
    make_train_test_val_division(dir_dataset_grouped_mp4, dir_dataset_grouped_ttv_mp4)

    print("****************************************************")
    print("****************************************************")

    # 3) Perform video data augmentation only on train set
    print("Data augmentation for train set")
    make_data_augmentation(dir_dataset_grouped_ttv_mp4, dir_dataset_grouped_augmented_ttv_mp4, "train", final_number=NUMBER_VIDEO_TRAIN)
    print("Data augmentation for val set")
    make_data_augmentation(dir_dataset_grouped_ttv_mp4, dir_dataset_grouped_augmented_ttv_mp4, "val", final_number=NUMBER_VIDEO_VAL)
    print("Data augmentation for test set")
    make_data_augmentation(dir_dataset_grouped_ttv_mp4, dir_dataset_grouped_augmented_ttv_mp4, "test", apply_augmentation=False)

    print("****************************************************")
    print("****************************************************")

    # 4) Create csv for train, val and test dataset
    class2label = create_dict_class2label(dir_dataset_grouped_augmented_ttv_mp4)
    print("class2label: ", class2label)

    print("----------------------------------")

    print("Create train csv")
    df_train = create_csv(dir_dataset_grouped_augmented_ttv_mp4, "train", class2label)
    df_train.to_csv(path_dataset_train_csv, index=False)
    print(df_train.info())

    print('---------------------------------')

    print("Create test csv")
    df_test = create_csv(dir_dataset_grouped_augmented_ttv_mp4, "test", class2label)
    df_test.to_csv(path_dataset_test_csv, index=False)
    print(df_test.info())

    print('---------------------------------')

    print("Create val csv")
    df_val = create_csv(dir_dataset_grouped_augmented_ttv_mp4, "val", class2label)
    df_val.to_csv(path_dataset_val_csv, index=False)
    print(df_val.info())

    print('---------------------------------')

    print("Number of train samples: ", len(df_train))
    print("Number of test samples: ", len(df_test))
    print("Number of val samples: ", len(df_val))
    print("Number of total samples: ", len(df_train) + len(df_val) + len(df_test))

    print("-----------------------------------")

    print("Train dataset unique classes")
    print(list(df_train['CLASS'].unique()))
    print("Number of train unique classes", len(list(df_train['CLASS'].unique())))

    print("-----------------------------------")

    print("Val dataset unique class")
    print(list(df_val['CLASS'].unique()))
    print("Number of val unique classes", len(list(df_val['CLASS'].unique())))

    print("-----------------------------------")

    print("Test dataset unique class")
    print(list(df_test['CLASS'].unique()))
    print("Number of test unique classes", len(list(df_test['CLASS'].unique())))

    print("-----------------------------------")

    print("Max NUM_FRAMES train: ", df_train['NUM_FRAMES'].max())
    print("Min NUM_FRAMES train: ", df_train['NUM_FRAMES'].min())
    print("Max NUM_FRAMES test: ", df_test['NUM_FRAMES'].max())
    print("Min NUM_FRAMES test: ", df_test['NUM_FRAMES'].min())
    print("Max NUM_FRAMES val: ", df_val['NUM_FRAMES'].max())
    print("Min NUM_FRAMES val: ", df_val['NUM_FRAMES'].min())