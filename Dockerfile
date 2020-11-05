FROM dorowu/ubuntu-desktop-lxde-vnc:bionic

RUN apt update && \
        DEBIAN_FRONTEND=noninteractive apt install -y \
        libxkbcommon-x11-0 wget locales gnupg2 lsb-release python3-pip python3-tk libcurl4-openssl-dev libssl-dev git && \
        locale-gen en_US en_US.UTF-8 && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 && export LANG=en_US.UTF-8 && \
        pip3 install virtualenv gdown && virtualenv -p python3 --system-site-packages env && \
        curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - && \
        sh -c 'echo "deb http://packages.ros.org/ros2/ubuntu `lsb_release -cs` main" > /etc/apt/sources.list.d/ros2-latest.list' && apt update && \
        DEBIAN_FRONTEND=noninteractive apt install -y ros-eloquent-ros-base python3-colcon-common-extensions ros-eloquent-cv-bridge

# Get data and VRep        
RUN wget https://www.coppeliarobotics.com/files/CoppeliaSim_Player_V4_0_0_Ubuntu18_04.tar.xz && \
    tar xvf CoppeliaSim_Player_V4_0_0_Ubuntu18_04.tar.xz && \
    rm CoppeliaSim_Player_V4_0_0_Ubuntu18_04.tar.xz && \
    gdown https://drive.google.com/uc?id=1hxHmeBEWxhaiIFYW4BKpatz_AFnmqNxt -O GDrive.tar.xz && \
    tar xf GDrive.tar.xz && \
    rm GDrive.tar.xz 

# Setup code
COPY . /root/Code/

# Setup Environment and python
RUN /bin/bash -c "source /root/env/bin/activate && source /opt/ros/eloquent/setup.sh && cd /root/Code/ros2 && colcon build --symlink-install" && \
    echo "export COPPELIASIM_ROOT=/root/CoppeliaSim_Player_V4_0_0_Ubuntu18_04" >> /root/.bashrc && \
    echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/CoppeliaSim_Player_V4_0_0_Ubuntu18_04" >> /root/.bashrc && \
    echo "export QT_QPA_PLATFORM_PLUGIN_PATH=/root/CoppeliaSim_Player_V4_0_0_Ubuntu18_04" >> /root/.bashrc && \
    echo "source /root/env/bin/activate" >> /root/.bashrc && \
    echo "source /opt/ros/eloquent/setup.sh" >> /root/.bashrc && \
    echo "source /root/Code/ros2/install/setup.sh" >> /root/.bashrc && \
    /bin/bash -c "source /root/env/bin/activate && pip install tensorflow==2.1.0 matplotlib hashids opencv-python==4.2.0.34 sklearn tensorflow-probability==0.9.0 pycurl cffi==1.11.5" && \
    git clone https://github.com/stepjam/PyRep.git

RUN cd PyRep && git checkout 7057e19c6f2dfb72d2ab101705fb534bf98aa722 && \
    /bin/bash -c "export COPPELIASIM_ROOT=/root/CoppeliaSim_Player_V4_0_0_Ubuntu18_04 && source /root/env/bin/activate && python3 setup.py install" 
