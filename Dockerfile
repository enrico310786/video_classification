FROM nvidia/cuda:11.4.2-runtime-ubuntu20.04

WORKDIR /app

COPY requirements_docker.txt /app/requirements_docker.txt
COPY get-pip.py /app/get-pip.py

RUN apt -y update && \
    apt-get -y upgrade && \
    apt-get install -y libsndfile1 && \
    apt-get install -y tzdata && \
    apt install --no-install-recommends -y build-essential software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt install --no-install-recommends -y python3.7 python3.7-distutils python3.7-dev && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN echo "installed python version"
RUN python3.7 --version

RUN python3.7 get-pip.py --user
RUN python3.7 -m pip --version

#RUN packages for opencv as https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

#https://stackoverflow.com/questions/66977227/could-not-load-dynamic-library-libcudnn-so-8-when-running-tensorflow-on-ubun
RUN apt-get install libcudnn8
RUN apt-get install libcudnn8-dev

#torch 1.12 CUDA 11.3
RUN python3.7 -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

#istall requirements
RUN python3.7 -m pip install -r requirements_docker.txt

#copy files python
COPY credentials /root/.aws/credentials
COPY download_s3.py /app/download_s3.py
COPY upload_s3.py /app/upload_s3.py
COPY logger.py /app/logger.py
COPY data.py /app/data.py
COPY utils.py /app/utils.py
COPY run_train_test_aws.py /app/run_train_test_aws.py
COPY train_test_classification_model.py /app/train_test_classification_model.py
COPY model.py /app/model.py

RUN python3.7 -m pip list

#comando per train e test
CMD ["python3.7","run_train_test_aws.py","--path_config_file","configs/video_classification_ucf_action_grouped_augmented.yaml","--aws_directory","ucf_action_sport/","--aws_bucket","questit-video-classification"]
