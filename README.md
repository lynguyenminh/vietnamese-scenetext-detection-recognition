# <center>Vietnamese Scenetext API</center>

## 1. Installation
```
git clone https://github.com/lynguyenminh/scenetext-api.git && cd scenetext-api
!pip install -r requirements.txt

cd parseq
pip install -r requirements.txt
pip install -e .
pip install torch==1.10.0 torchtext==0.11.0
```


## 2. Run 

```
cd scenetext-api && streamlit run Home.py
```

I use the following links to fix `bugs` encountered during implementation: [error_fix.txt](error_fix.txt)

## 3. References
[1]. https://bamblebam.medium.com/how-to-deploy-your-machine-learning-model-using-streamlit-925368b266ad