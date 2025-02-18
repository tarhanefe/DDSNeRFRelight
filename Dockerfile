FROM nvcr.io/nvidia/pytorch:23.05-py3
MAINTAINER Ehsan Pajouheshgar<ehsan.pajouheshgar@epfl.ch>

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update &&  TZ="Europe/Zurich" apt-get install -y \
    curl vim htop\
    ca-certificates \
    cmake \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    zip \
    unzip ssh \
    tmux \
    ffmpeg \
 && rm -rf /var/lib/apt/lists/*



RUN pip3 install jupyter jupyterlab pandas \
                 matplotlib Pillow h5py \
                 tqdm moviepy scipy pyyaml\
                 opencv-contrib-python-headless

RUN pip3 install -U torch \
    torchvision \
    torchaudio --index-url https://download.pytorch.org/whl/cu117

RUN jupyter nbextension enable --py widgetsnbextension


USER root
RUN mkdir /opt/lab
COPY setup.sh /opt/lab/
RUN chmod -R a+x /opt/lab/


RUN apt-get update && apt-get install -y openssh-server
RUN mkdir /var/run/sshd
RUN echo 'root:root' | chpasswd
RUN sed -i 's/#*PermitRootLogin prohibit-password/PermitRootLogin yes/g' /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sed -i 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' /etc/pam.d/sshd

ENV NOTVISIBLE="in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]
