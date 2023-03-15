# <center>Vietnamese Scenetext API</center>

**Demo:**

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/mUAIIVRsxvM/0.jpg)](https://www.youtube.com/watch?v=mUAIIVRsxvM)

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
