# <center>Vietnamese Scenetext API</center>

**Demo:**

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/mUAIIVRsxvM/0.jpg)](https://www.youtube.com/watch?v=mUAIIVRsxvM)

## 1. Installation
### 1. Using docker (Recommended)


### 1.2. Using in virtualenv or local (Not Recommended)
```
git clone https://github.com/lynguyenminh/vietnamese-scenetext-detection-recognition-api.git && cd vietnamese-scenetext-detection-recognition-api
!pip install -r requirements.txt

cd parseq
pip install -r requirements.txt
pip install -e .
pip install torch==1.10.0 torchtext==0.11.0
```


## 2. Run 

```
cd vietnamese-scenetext-detection-recognition-api && streamlit run Home.py
```


## 3. References
[1]. https://bamblebam.medium.com/how-to-deploy-your-machine-learning-model-using-streamlit-925368b266ad
