# 农作物病害检测

18年10月做的比赛，是我的起步hh，向大佬学习。

微调**VGG19**和**Xception**并模型融合 , 解决 AI Challenger 2018 农作物病害检测问题。

## 依赖

- [NumPy](http://docs.scipy.org/doc/numpy-1.10.1/user/install.html)
- [Tensorflow](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html)
- [Keras](https://keras.io/#installation)
- [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/)

## 介绍

```
PLANTDIEASE
│  augmentor.py # 数据增强
│  config.py  # 配置文件
│  data_generator.py # 自定义数据生成器
│  fuck.md
│  fuck_model.py # 模型
│  ronghe_submit.py # 生成结果
│  submit.json # 生成结果
│  train.py # 训练
│  utils.py # 辅助
│
├─ai_challenger_pdr2018_testA_20180905 #存放测试集A
├─ai_challenger_pdr2018_trainingset_20180905 #存放训练集
├─ai_challenger_pdr2018_validationset_20180905 #存放验证集
├─logs 
└─models #保存模型权重
```

## 使用

- 首先点此，[下载数据](https://challenger.ai/dataset/pdd2018)
- 修改`config.py`，指定训练集，验证集和测试集
- 运行`train.py`进行融合模型的训练
- 运行`ronghe_submit.py`获得测试集结果`submit.json`

## Reference

- [Crop-Disease-Detection](https://github.com/foamliu/Crop-Disease-Detection)
- [Keras模型融合](<https://blog.csdn.net/qq_33266320/article/details/82558740>)

# TODO

- 整理之后农作物病虫害前端代码
- 整理之后农作物病虫害后端代码
- 学习之后农作物病虫害安卓端代码