# 结构说明

## data

- 用来存储数据
- `BTCUSDT.csv`是原始的K线数据，由于csv文件过大，可以通过云盘链接下载之后存放至本地目录 `https://cloud.tsinghua.edu.cn/d/883823d4daa24a8ab91e/`

## data_process

- 用来处理数据的脚本

## model

- 里面有data_loader、超参数设置的脚本、模型设置的脚本、训练的脚本

## paper

- 里面有proposal、以及记录探索过程的markdown文件、以及依赖的图片、相关的论文

## log

- 记录了每一次的log输出、混淆矩阵图片、以及相应的模型文件

## reuslt

- 目前没有什么用

# 流程

- 先去链接下载基础数据
- 用 `data_process/feature_derivation.py`生成标签
- 用 `data_process/split_data.py`来切分训练集、验证集、测试集
- 之后理应点击 `model/train.py`就可以开始训练目前的模型了
