# This is an auto generated Dockerfile for ros:ros-core
# generated from docker_images/create_ros_core_image.Dockerfile.em
FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

# setup timezone
RUN echo 'Etc/UTC' > /etc/timezone && \
    ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    apt-get update && \
    apt-get install -q -y --no-install-recommends tzdata && \
    rm -rf /var/lib/apt/lists/*

# install packages
RUN apt-get update && apt-get install -q -y --no-install-recommends \
    dirmngr \
    gnupg2 \
    && rm -rf /var/lib/apt/lists/*

# setup keys
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# setup sources.list
RUN echo "deb http://packages.ros.org/ros/ubuntu bionic main" > /etc/apt/sources.list.d/ros1-latest.list

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

ENV ROS_DISTRO melodic

# install ros packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-melodic-ros-core=1.4.1-0* ros-melodic-cv-bridge \
    && rm -rf /var/lib/apt/lists/*

# setup entrypoint
COPY ./ros_entrypoint.sh /
RUN chmod +x ros_entrypoint.sh
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]

RUN apt-get update \
 && apt-get install -y vim wget tmux less htop python-pip python-tk\
 libsm6 libxext6 libxrender-dev \
 && apt-get clean

RUN sudo pip install --no-cache-dir numpy==1.16.2 matplotlib==2.2.5 scipy==1.2.3 
RUN sudo pip install --no-cache-dir torch==1.4.0 opencv-python==4.2.0.32 future
# having trouble installing cupy with sudo 
RUN pip install --no-cache-dir cupy==6.7.0