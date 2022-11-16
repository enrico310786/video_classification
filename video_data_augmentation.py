import cv2
import albumentations as A
import skvideo.io
import os

from utils import load_video, create_augmented_video

path_video = "video_examples"
name_video = "132.mp4"
numero_ripetizioni = 20

'''
transform = A.ReplayCompose([
    #A.ElasticTransform(alpha=0.1, p=0.5),
    A.GridDistortion(distort_limit=0.4, p=0.5),
    ##A.OpticalDistortion(distort_limit=0.5, p=1),
    #A.ShiftScaleRotate(scale_limit=0.05, rotate_limit=5, p=0.5),
    A.Rotate(limit=5, p=0.5),
    ##A.GaussNoise(var_limit=[30.0, 70.0], mean=1, p=1),
    A.RGBShift(r_shift_limit=50, g_shift_limit=50, b_shift_limit=50, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.CLAHE(p=0.5),
    A.PixelDropout(drop_value=0, dropout_prob=0.02, p=0.5),
    A.PixelDropout(drop_value=255, dropout_prob=0.02, p=0.5),
    A.Blur(blur_limit=(2, 4), p=0.5)
])
'''

transform = A.ReplayCompose([
    A.ElasticTransform(alpha=0.5, p=0.5),
    A.ShiftScaleRotate(scale_limit=0.05, rotate_limit=10, p=0.5),
    A.RGBShift(r_shift_limit=50, g_shift_limit=50, b_shift_limit=50, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.CLAHE(p=0.5),
    A.PixelDropout(drop_value=0, dropout_prob=0.01, p=0.5),
    A.PixelDropout(drop_value=255, dropout_prob=0.01, p=0.5),
    A.Blur(blur_limit=(2, 4), p=0.5)
])

def augment_frames(frame_list):

    data = None
    augmented_frame_list = []

    for i, item in enumerate(frame_list):

        if i == 0:
            first_image = cv2.cvtColor(item, cv2.COLOR_BGR2RGB)
            data = transform(image=first_image)
            new_image = data['image']
        else:
            image = cv2.cvtColor(item, cv2.COLOR_BGR2RGB)
            new_image = A.ReplayCompose.replay(data['replay'], image=image)['image']

        augmented_frame_list.append(new_image)

    return augmented_frame_list


######################################################################
# Execution
######################################################################
# 1: load the video and split it into frames
path_original_video = os.path.join(path_video, name_video)
frame_list, _, fps, _ = load_video(path_original_video)
print("len(frame_list): ", len(frame_list))
print("width: ", frame_list[0].shape[0])
print("height: ", frame_list[0].shape[1])

width = frame_list[0].shape[0]
height = frame_list[0].shape[1]
print("frame_list[0].shape", frame_list[0].shape)

for i in range(numero_ripetizioni):
    # 2: create the augmented frames
    augmented_frame_list = augment_frames(frame_list)

    # 3: crete new video with the augmented frames
    name_augmented_video = name_video.split(".")[0] + "_" + str(i) + ".mp4"
    path_augmented_video = os.path.join(path_video, name_augmented_video)
    create_augmented_video(augmented_frame_list, path_augmented_video, fps)