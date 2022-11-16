"""
Example of loading a clip video and make inference with the trained model
"""

import yaml
import torch
from torchsummary import summary
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample
)

##############################
# CONSTANTS AND METHODS
###############################
from model import VideoClassificationModel

video_path = "/mnt/disco_esterno/ucf_sports_actions/ucf_action_grouped_ttv_mp4/test/Kicking/6063-21_70056.mp4"
path_last_checkpoint = "exps_ucf_action/ucf_action_grouped_augmented_v2/models_augmented_v2/best.pth"
path_config_file = "configs/video_classification_ucf_action_grouped_augmented.yaml"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: ", device)

lista_classes = ['Diving-Side', 'Golf-Swing', 'Kicking', 'Lifting', 'Riding-Horse', 'Run-Side', 'SkateBoarding-Front', 'Swing-Bench', 'Swing-SideAngle', 'Walk-Front']
class2label = {k: v for (v, k) in enumerate(lista_classes)}
label2class = {k: v for (v, k) in class2label.items()}

side_size = 256
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
crop_size = 256
num_frames = 64


transform = ApplyTransformToKey(
    key="video",
    transform=Compose(
        [
            UniformTemporalSubsample(num_frames),
            Lambda(lambda x: x / 255.0),
            NormalizeVideo(mean, std),
            ShortSideScale(size=side_size),
            CenterCropVideo(crop_size=(crop_size, crop_size))
        ]
    ),
)

def load_config(path="configs/default.yaml") -> dict:
    """
    Loads and parses a YAML configuration file.

    :param path: path to YAML configuration file
    :return: configuration dictionary
    """
    with open(path, "r", encoding="utf-8") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg


##############################
# LOAD MODEL
###############################

print('path_config_file: ', path_config_file)
cfg = load_config(path_config_file)

print("Load the model")
model = VideoClassificationModel(cfg).to(device)

print("Load the best checkpoint: ", path_last_checkpoint)
checkpoint = torch.load(path_last_checkpoint, map_location=torch.device(device))
model.load_state_dict(checkpoint['model'])
model = model.eval()
model = model.to(device)

summary(model, input_size=(3, 32, 256, 256))




##############################
# INFERENCE
###############################

video = EncodedVideo.from_path(video_path)
start_time = 0
# follow this post for clip duration https://towardsdatascience.com/using-pytorchvideo-for-efficient-video-understanding-24d3cd99bc3c
clip_duration = int(video.duration)
# print("clip_duration: ", clip_duration)
end_sec = start_time + clip_duration
video_data = video.get_clip(start_sec=start_time, end_sec=end_sec)
video_data = transform(video_data)
inputs = video_data["video"]
inputs = inputs[None].to(device)

print("type(inputs): ", type(inputs))
print("inputs.size(): ", inputs.size())

with torch.no_grad():
    preds_pre_act = model(inputs)
    post_act = torch.nn.Softmax(dim=1)
    preds = post_act(preds_pre_act)

    pred_values, pred_label = torch.max(preds, 1)
    pred_label = pred_label.cpu().numpy().tolist()[0]
    pred_values = round(pred_values.cpu().numpy().tolist()[0], 3)

    print("Predicted class: ", label2class[pred_label])
    print("Score of predicted class: ", pred_values)