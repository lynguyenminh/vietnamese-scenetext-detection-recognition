FROM nvidia/cuda:10.2-devel-ubuntu18.04

ENV PATH="/root/anaconda3/bin:$PATH" FORCE_CUDA="1" \
    TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
    
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC

RUN apt-get update --allow-releaseinfo-change && apt-get install -y libgbm-dev -y \
    software-properties-common dirmngr -y \
    build-essential -y \
    libgl1-mesa-glx libxrender1 libfontconfig1 -y \
    libglib2.0-0 -y \
    libsm6 libxext6 libxrender-dev -y \
    vim zip unzip wget screen -y \
    gnupg2 -y \
    libgl1-mesa-glx -y \
    git libmagickwand-dev -y

RUN cd /root && wget https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh \
    && cd /root && bash Anaconda3-2020.07-Linux-x86_64.sh -b -p ./anaconda3 \
    && bash -c "source /root/anaconda3/etc/profile.d/conda.sh"



# Thiết lập thư mục làm việc
WORKDIR /craft-api

# Sao chép các tệp cần thiết vào thư mục làm việc
COPY . /craft-api

# Cài đặt các gói phụ thuộc
RUN pip install --no-cache-dir -r requirements.txt

# Mở cổng cho dịch vụ
EXPOSE 9001

# # Khởi động ứng dụng
# CMD ["python", "app.py"]
