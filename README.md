# D2-MetricLearning

When dealing with a large range of categories like face recognition or 
landmark recognition, metric learning become a popular option to the problem.

This demo project is an extension of [detectron2](https://github.com/facebookresearch/detectron2).

[Set up environment with docker](docker/quick_start.md)

# Data

You may use kaggle API to download the data:
```bash
# make sure `kaggle.json` is copied to /.kaggle/kaggle.json
. script/fectch_data.sh
```
The dataset is a 100GB zip file and may take a while to download and unzip.

Data should be preprocessed/split by this [script](preprocess/preprocess.py)

The code will get rid of categories with samples < 15 images.

Eventually it will left 27756 categories and 1223195 samples. 
The original kaggle dataset contains 1.6M images

By doing that we simplify the problems and concentrate on the basic model building first.

# Previous Work and Review

1. [Kaggle Landmark 1st place solution](https://github.com/ruodingt/kaggle-landmark-recognition-2020-1st-place)
2. [Kaggle Landmark 2nd place solution](https://github.com/ruodingt/instance_level_recognition)
3. [Kaggle Landmark 3rd place solution](https://github.com/ruodingt/Google-Landmark-Recognition-2020-3rd-Place-Solution)


They all have something in common:
1. In terms of model architecture all of the top three more or less used similar solution:
`Backbone + Bottleneck Layer (512) + ArcMarginHead + CE-Loss/Focal-Loss`. 
2. All of them stressed the importance of `post-processing`, which could bring a lot of performance gain.
3. They all demonstrated that global feature is good enough without attention to local features
4. They all used ensembling method to boost performance but it also worth noticing that it is not 
a significant boost considering the amount of computing power involved
5. Among the top3 places solutions, all of them used ArcMargin to build the loss function. 
It was originally used in face recognition.
6. Using gradient accumulation to increase the effective batch-size. 
Though it is not yet clear how would this affect batchnorm
7. FP16 Mixed precision training with apex.


# Challenges
The biggest challenges of this specific problem came from the loss function itself.

It seems the arcface tends to get stuck at local optimum.

Here is some discussion around the training of `ArcMargin`:
1. https://www.kaggle.com/c/recursion-cellular-image-classification/discussion/109987


# Experiment and Results
Due to limited time committed and computing power, I haven't be able to produce any meaningful
result at this stage.

Our default setting only includes `NUM_CLASSES: 27756` categories instead of all the 80K.
We filtered out categories in the `train.csv` which have less than 15 images/category.

With [exp7-res50.yaml](vit_metric/config/exp7-res50.yaml), 
It only gets `0.04` for acc (0-1) and `0.006` for gap(0-1) after one epoch.

Typical training schedule recommended by top Kaggler require 10 epoch, 
which can take a few days without distributed training.

```bash
cd vit_metric
python train_net.py --config-file config/exp11-res50.yaml 
```

## Loss Function
[Loss Explain](https://www.groundai.com/project/arcface-additive-angular-margin-loss-for-deep-face-recognition/1)

Stealing Loss function implementation from:

https://github.com/foamliu/InsightFace-v2/blob/e7b6142875f3f6c65ce97dd1b2b58156c5f81a3d/models.py#L327

https://github.com/ronghuaiyang/arcface-pytorch/blob/47ace80b128042cd8d2efd408f55c5a3e156b032/models/metrics.py#L10


