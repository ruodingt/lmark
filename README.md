# D2-MetricLearning

When dealing with a large range of categories like face recognition or 
landmark recognition, metric learning become a popular option to the problem.

This demo project is an extension of [detectron2](https://github.com/facebookresearch/detectron2).

Set up environment with docker



# Data

You may use kaggle API to download the data:
```bash
# make sure `kaggle.json` is copied to /.kaggle/kaggle.json
. script/fectch_data.sh
```
The dataset is a 100GB zip file and may take a while to download and unzip

CFG:
https://github.com/psinger/kaggle-landmark-recognition-2020-1st-place/blob/3d39374f261de1762cddf83cec0d2f7b35efde45/configs/config4.py

# Previous Work and Review
Among the top3 places solutions
ArcMargin 


# Experiment and Results



# Reference
1. [Kaggle Landmark 1st place solution]()
1. [Kaggle Landmark 2nd place solution]()
1. [Kaggle Landmark 3rd place solution]()

## Loss Function
Stealing Loss function implementation from:
https://github.com/foamliu/InsightFace-v2/blob/e7b6142875f3f6c65ce97dd1b2b58156c5f81a3d/models.py#L327

https://github.com/ronghuaiyang/arcface-pytorch/blob/47ace80b128042cd8d2efd408f55c5a3e156b032/models/metrics.py#L10
