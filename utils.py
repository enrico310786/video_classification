import os
import random
import torch
import numpy as np
import torch.distributed as dist
import matplotlib.pyplot as plt
import cv2
from pytorchvideo.data.encoded_video import EncodedVideo
import skvideo.io
import albumentations as A

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def setup(rank, cfg):
    """Initializes a distributed training process group.

    Args:
        rank: a unique identifier for the process
        cfg: config dict
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '5000'
    #dist.init_process_group(backend=cfg["backend"],
    #                        rank=rank,
    #                        world_size=cfg["n_gpus"],
    #                        init_method='env://')

    dist.init_process_group(backend=cfg["backend"],
                            rank=rank,
                            world_size=cfg["n_gpus"])


def find_last_checkpoint_file(checkpoint_dir, use_best_checkpoint=False):
    '''
    Cerco nella directory checkpoint_dir il file .pth con l'epoca maggiore.
    Se use_best_checkpoint = True prendo il best checkpoint
    Se use_best_checkpoint = False prendo quello con l'epoca maggiore tra i checkpoint ordinari
    :param checkpoint_dir:
    :param use_best_checkpoint:
    :return:
    '''
    print("Cerco il file .pth in checkpoint_dir {}: ".format(checkpoint_dir))
    list_file_paths = []

    for file in os.listdir(checkpoint_dir):
        if file.endswith(".pth"):
            path_file = os.path.join(checkpoint_dir, file)
            list_file_paths.append(path_file)
            print("Find: ", path_file)

    print("Number of files .pth: {}".format(int(len(list_file_paths))))
    path_checkpoint = None

    if len(list_file_paths) > 0:
        if use_best_checkpoint:
            if os.path.isfile(os.path.join(checkpoint_dir, 'best.pth')):
                path_checkpoint = os.path.join(checkpoint_dir, 'best.pth')
        else:
            list_epoch_number = []
            for path in list_file_paths:
                file_name = path.split('/')[-1]
                file_name_no_extension = file_name.split('.')[0]

                if file_name_no_extension.split('_')[0] == "checkpoint":
                    number_epoch = int(file_name_no_extension.split('_')[1])
                else:
                    continue
                list_epoch_number.append(number_epoch)
            max_epoch = max(list_epoch_number)
            path_checkpoint = os.path.join(checkpoint_dir, 'checkpoint_' + str(max_epoch) + '.pth')

    return path_checkpoint


def save_checkpoint(model, rank, epoch, optimizer, lr_scheduler, best_val_epoch_accuracy, checkpoint_dir, best):
    """Saves the model in master process and loads it everywhere else.

    Args:
        model: the model to save
        gpu: the device identifier
        epoch: the training epoch
    Returns:
        model: the loaded model
    """
    path_ckp = None
    if rank == 0:
        # All processes should see same parameters as they all start from same
        # random parameters and gradients are synchronized in backward passes.
        # Therefore, saving it in one process is sufficient.

        save_obj = {
            'model': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'best_eva_accuracy': best_val_epoch_accuracy
        }

        if best:
            print("Save best checkpoint at: ", os.path.join(checkpoint_dir, 'best.pth'))
            torch.save(save_obj, os.path.join(checkpoint_dir, 'best.pth'), _use_new_zipfile_serialization=False)
            print("Save checkpoint at: ", os.path.join(checkpoint_dir, 'checkpoint_' + str(epoch) + '.pth'))
            torch.save(save_obj, os.path.join(checkpoint_dir, 'checkpoint_' + str(epoch) + '.pth'), _use_new_zipfile_serialization=False)
        else:
            print("Save checkpoint at: ", os.path.join(checkpoint_dir, 'checkpoint_' + str(epoch) + '.pth'))
            torch.save(save_obj, os.path.join(checkpoint_dir, 'checkpoint_' + str(epoch) + '.pth'), _use_new_zipfile_serialization=False)

    # use a barrier() to make sure that process 1 loads the model after process
    # 0 saves it.
    dist.barrier()
    # configure map_location properly
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    checkpoint = torch.load(path_ckp, map_location=map_location)
    model.module.load_state_dict(checkpoint['model'])


def cleanup():
    dist.destroy_process_group()


def plot_learning_curves(epochs, train_losses, val_losses, train_accuracies, val_accuracies, path_save):
    '''
    La funzione plotta le learning curves sul train e validation set di modelli già allenati
    '''
    x_axis = range(0, epochs)

    plt.figure(figsize=(27,9))
    plt.suptitle('Learning curves ', fontsize=18)
    #primo plot
    plt.subplot(121)
    plt.plot(x_axis, train_losses, label='Training Loss')
    plt.plot(x_axis, val_losses, label='Validation Loss')
    plt.legend()
    plt.title('Train and Validation Losses', fontsize=16)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Loss', fontsize=16)

    #secondo plot
    plt.subplot(122)
    plt.plot(x_axis, train_accuracies, label='Training Accuracy')
    plt.plot(x_axis, val_accuracies, label='Validation Accuracy')
    plt.legend()
    plt.title('Train and Validation accuracy', fontsize=16)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)

    plt.savefig(os.path.join(path_save, "learning_curves.png"))


def load_video(video_path):

    frame_list = []
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        frame_list.append(frame)
    cap.release()

    video = EncodedVideo.from_path(video_path)
    return frame_list, len(frame_list), fps, int(video.duration)


def create_augmented_video(frame_list, path_augmented_video, fps):

    writer = skvideo.io.FFmpegWriter(path_augmented_video,
                                     inputdict={'-r': str(fps)},
                                     outputdict={'-r': str(fps), '-c:v': 'libx264', '-preset': 'ultrafast', '-pix_fmt': 'yuv444p'})

    for i, image in enumerate(frame_list):
        image = image.astype('uint8')
        writer.writeFrame(image)

    # close writer
    writer.close()


def subsample_and_resize_frames(list_frames, sample_rate, img_size):
    subsampled_list_frames = []
    for i, frame in enumerate(list_frames):
        if i % sample_rate == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # ripasso il frame in RGB
            frame = cv2.resize(frame, (img_size, img_size))
            subsampled_list_frames.append(frame)
    return subsampled_list_frames


transform = A.ReplayCompose([
    #A.ElasticTransform(alpha=0.1, p=0.5),
    A.GridDistortion(distort_limit=0.4, p=0.6),
    ##A.OpticalDistortion(distort_limit=0.5, p=1),
    #A.ShiftScaleRotate(scale_limit=0.05, rotate_limit=5, p=0.5),
    A.Rotate(limit=5, p=0.6),
    ##A.GaussNoise(var_limit=[30.0, 70.0], mean=1, p=1),
    A.RGBShift(r_shift_limit=50, g_shift_limit=50, b_shift_limit=50, p=0.6),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
    A.CLAHE(p=0.6),
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
'''

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

        #new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR): the images have to output as RGB images
        augmented_frame_list.append(new_image)

    return augmented_frame_list