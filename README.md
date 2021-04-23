# Diabetic Retinopathy Detection 
This Repo contains my scripts for the [EEL 6825: Pattern Recogntion](http://www.wu.ece.ufl.edu/courses/eel6825s21/) Spring 2021 Course.

## Papers
- [Skin lesion classification with ensemble of squeeze-and-excitation networks and semi-supervised learning](https://arxiv.org/abs/1809.02568)
- [Self-training with Noisy Student improves ImageNet classification](https://arxiv.org/pdf/1911.04252.pdf)
- [Bottleneck Transformers for Visual Recognition](https://arxiv.org/pdf/2101.11605.pdf)
- [Rotate to Attend: Convolutional Triplet Attention Module](https://arxiv.org/pdf/2010.03045.pdf)

## Features
- &#x2611; Balanced Sampler 

- &#x2611; Mixed Precision

- &#x2611; Gradient Accumulation  

- &#x2611; Model freeze-unfreeze

- &#x2611; Optimum Learning Rate Finder

- &#x2611; TTA 


## Resources
- [Margin Focal Loss](https://www.kaggle.com/c/siim-isic-DR-classification/discussion/155201)
- [APTOS Gold Medal Solutions](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108307): Although data type is different but it might be helpful.
- [DR Recognition via Visual Attention](https://github.com/SaoYan/IPMI2019-AttnMel)

## Can be useful
- [Deep Metric Learning Solution For MVTec Anomaly Detection Dataset](https://medium.com/analytics-vidhya/spotting-defects-deep-metric-learning-solution-for-mvtec-anomaly-detection-dataset-c77691beb1eb)
- [Ugly Duckling Concept](https://www.kaggle.com/c/siim-isic-DR-classification/discussion/155348)
- Humpback Whale Classification 1st place [solution](https://www.kaggle.com/c/humpback-whale-identification/discussion/82366)
- [Attention model](https://www.kaggle.com/c/siim-isic-DR-classification/discussion/171745) for feature extraction: Scoring `0.9287` with Resnet only. 

## How to run
- Run `git clone https://github.com/tahsin314/Diabetic_Retinopathy-Detection-Pattern-Recognition-Course-Project`
- In the `dr_config.ini` file change the `data_dir` variable to your data directory name.
- Run `conda env create -f environment.yml`
- Run `train.py`. Change parameters according to your preferences from the `dr_config.ini` file before training.

### One important thing about EfficientNet
EfficientNet's are designed to take in to account input image dimensions.

So if you want to squeeze every last droplet from your model make sure to use same image resolutions as described below:

```
Efficientnet-B0 : 224
Efficientnet-B1 : 240
Efficientnet-B2 : 260
Efficientnet-B3 : 300
Efficientnet-B4 : 380
Efficientnet-B5 : 456
Efficientnet-B6 : 528
Efficientnet-B7 : 600
```

## Issues
- Currently muli gpu training only works with `distributed_backend='dp'`. However, `Stochastic Weight Averaging` breaks it. 
- `distributed_backend='dp'` also fails to gather data from multiple GPUs and returns predictions and labels with shape `(batch_num, )` where it is supposed to return data with shape `(num_samples, )` (`num_samples ≈ num_batches * batch_size`). This is probably a *pytorch-lightning* bug. It is advised to comment out `distributed_backend='dp'` and training on single GPU at this moment. 

**Alternative:** Try running inference on test data on single GPU or set `batch_size=1` during test.
### Update [20/04/2021]
Added an ugly hack for gathering predictions and labels with `distributed_backend='dp'`. Now data can be gathered from multiple GPUs. Hope that *pytorch-Lightning* will fix their `gather` issue.