import torch
import os
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms import Compose, Lambda, Resize
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample
)
# https://pytorch.org/hub/facebookresearch_pytorchvideo_resnet/

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader


class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, df_dataset, cfg, dataset_path) -> None:
        super().__init__()

        self.df_dataset = df_dataset
        self.dataset_path = dataset_path
        self.cfg = cfg
        side_size = 256
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        crop_size = 256
        num_frames = cfg["num_frames"]
        #sampling_rate = cfg["sampling_rate"]
        #frames_per_second = cfg["frames_per_second"]
        do_crop_video = cfg.get("crop_video", 1.0) > 0.0

        if do_crop_video:
            print("do_crop_video TRUE")
            self.transform = ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(num_frames),
                        Lambda(lambda x: x/255.0),
                        NormalizeVideo(mean, std),
                        ShortSideScale(size=side_size),
                        CenterCropVideo(crop_size=(crop_size, crop_size))
                    ]
                ),
            )
        else:
            print("do_crop_video FALSE")
            self.transform = ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(num_frames),
                        Lambda(lambda x: x/255.0),
                        NormalizeVideo(mean, std),
                        #Resize((crop_size, crop_size))
                        #ShortSideScale(size=side_size)
                    ]
                ),
            )

    def __len__(self):
        return len(self.df_dataset)

    def __getitem__(self, idx):
        video_path = os.path.join(self.dataset_path, self.df_dataset.iloc[idx]["PATH"])
        label = self.df_dataset.iloc[idx]["LABEL"]

        if not os.path.exists(video_path):
            print("The video '{}' does not exist ".format(video_path))

        video = EncodedVideo.from_path(video_path)
        start_time = 0
        # follow this post for clip duration https://towardsdatascience.com/using-pytorchvideo-for-efficient-video-understanding-24d3cd99bc3c
        clip_duration = int(video.duration)
        #print("clip_duration: ", clip_duration)
        end_sec = start_time + clip_duration
        video_data = video.get_clip(start_sec=start_time, end_sec=end_sec)
        video_data = self.transform(video_data)
        inputs = video_data["video"]

        return inputs, label


def create_loaders(df_dataset_train, df_dataset_val, df_dataset_test, cfg):

    dataset_path = cfg['dataset_path']
    batch_size = cfg['batch_size']

    # 1 - I instatiate the dataset class of train, val and test sets
    classification_dataset_test = None
    classification_dataset_train = ClassificationDataset(df_dataset=df_dataset_train, cfg=cfg, dataset_path=dataset_path)
    classification_dataset_val = ClassificationDataset(df_dataset=df_dataset_val, cfg=cfg, dataset_path=dataset_path)
    if df_dataset_test is not None:
        classification_dataset_test = ClassificationDataset(df_dataset=df_dataset_test, cfg=cfg, dataset_path=dataset_path)


    # 2 - I instatiate the dataloader
    classification_dataloader_train = DataLoader(dataset=classification_dataset_train,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 num_workers=0,
                                                 drop_last=False)

    classification_dataloader_val = DataLoader(dataset=classification_dataset_val,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0,
                                               drop_last=False)

    classification_dataloader_test = None
    if classification_dataset_test is not None:
        classification_dataloader_test = DataLoader(dataset=classification_dataset_test,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=0,
                                                    drop_last=False)

    return classification_dataloader_train, classification_dataloader_val, classification_dataloader_test