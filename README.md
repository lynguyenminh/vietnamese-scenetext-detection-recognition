# <center>Vietnamese Scenetext API</center>

## 0. Introduction: 
Tôi đã phát triển một ứng dụng nhỏ nhằm giải quyết bài toán scenetext trong ngôn ngữ Tiếng Việt. Scene text là một phần của OCR, tập trung vào việc xử lý ảnh chứa các đoạn văn bản xuất hiện trên các biển hiệu, biến báo giao thông và nhiều nguồn dữ liệu khác. Đây là một bài toán đầy thách thức, đặc biệt là do sự nhiễu từ môi trường, phông chữ đa dạng và nền ảnh đa dạng.

Với kinh nghiệm tham gia nhiều cuộc thi liên quan đến scene text và đạt được vị trí đáng kể trong những cuộc thi đó, tôi đã tổng hợp mã nguồn vào repository này để tạo thành một ứng dụng demo nhỏ. Trong quá trình thi, chúng tôi thường chạy ứng dụng trên CLI để tạo ra kết quả nhanh chóng. Tuy nhiên, trong repository này, tôi đã tạo một giao diện người dùng trực quan để hỗ trợ việc trực quan hóa dữ liệu sau khi dự đoán, nhằm tìm ra các điểm yếu của mô hình.

**Video Demo:**

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/mUAIIVRsxvM/0.jpg)](https://www.youtube.com/watch?v=mUAIIVRsxvM)

Mô hình bài toán
![image](https://github.com/lynguyenminh/vietnamese-scenetext-detection-recognition/assets/82688630/629258f0-210e-4621-afbf-4270df96bdd1)


## 1. Installation
### 1. Using docker (Recommended)
Pull docker images: 
```
docker pull lynguyenminh/scenetext-api:v1
```

Git clone source code: 
```
git clone https://github.com/lynguyenminh/vietnamese-scenetext-detection-recognition-api.git
```

Run container: 
```
docker run -it --name scenetext-api-v1 -p 5000:5000 -v ./vietnamese-scenetext-detection-recognition-api/:/vietnamese-scenetext-detection-recognition-api lynguyenminh/scenetext-api:v1
```
Download weights: 
```
cd vietnamese-scenetext-detection-recognition-api
sh download_models.sh
```


### 1.2. Using in virtualenv or local (Not Recommended)
Download source code and install environment: 
```
git clone https://github.com/lynguyenminh/vietnamese-scenetext-detection-recognition-api.git
cd vietnamese-scenetext-detection-recognition-api
sh download_models.sh
pip install -r requirements.txt

cd parseq
pip install -r requirements.txt
pip install -e .
pip install torch==1.10.0 torchtext==0.11.0
```


## 2. Run application

```
streamlit run 01_Home.py --server.port 5000
```


## 3. References
[1]. https://bamblebam.medium.com/how-to-deploy-your-machine-learning-model-using-streamlit-925368b266ad
