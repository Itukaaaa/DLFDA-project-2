**注意：代码文件经反复修改和复用，导致有些功能的实现可能需要对代码进行一定的调整，比如更改一些硬编码的文件名、取消一些注释等等。**

下面介绍每个文件夹，和重要代码的功能。

## backtest

- images/、logs/、multi_images/、threshold_images/四个文件夹存储回测得到的结果。
- backtest.md记录了一些中间信息。
- `python test.py`和`python multi_test.py`分别进行单模型回测和双模型投票回测。模型信息硬编码入代码，使用时需调整。
- `python find_best_threshold.py`和`python multi_find_best_threshold.py`分别用来遍历搜索单模型阈值和双模型投票阈值。模型信息硬编码入代码，使用时需调整。

## data

- 用来存储数据。
- `BTCUSDT.csv`是原始的K线数据，由于csv文件过大，可以通过云盘链接下载之后存放至本地目录 `https://cloud.tsinghua.edu.cn/d/883823d4daa24a8ab91e/`。

## data_process

- 用来处理数据的脚本。
- `python data_process/data_cut.py`会将数据切分成一部分的训练时数据集（包括训练、验证、测试集），和另一部分的推理集。推理集会直接储存在splits文件夹中，而训练时数据集会在训练时被分割成三份并存储至splits文件夹。使用时需要调整注释。
- `python data_process/data_to_bin.py`是用来产生二分类数据集的。实际上，由于二分类训练效果不好，所以这部分后期没有维护。
- `python data_process/feature_1.py`是用来给原始数据计算特征的。
- `python data_process/feature_derivation.py`是用来计算标签的。两种标签计算方式需要手动调节create_label函数内容。
- 其余代码都是分析用到的。

## Denhance

- 这个文件夹是分析数据特征，并构造新的数据集用的。

## infer_result

- 用来存储推理之后的中间结果的辅助文件夹。中间结果是带有标签、以及模型输出结果的csv文件。

## logs

- 用来存储全部的log。log由创建时间命名。

## model

- `config.py`中调整模型参数。本次作业中实际未进行调参工作。
- `python model/binary_main.py --csv <训练时数据集.csv>`进行二分类器训练。
- `python model/main.py --csv <训练时数据集.csv>`进行纯交叉熵三分类器训练。
- `python model/cm_main.py --csv <训练时数据集.csv>`进行混合损失（交叉熵+软混淆目标）三分类器训练。
- `python model/inference.py --csv <推理回测集.csv> --ckpt <模型.pt> --nclasses <2或3表示分类数量> --outfile <推理中间文件.csv>`进行推理。推理后调用前述回测文件进行回测。

## paper

- 里面有proposal、以及记录探索过程的markdown文件、以及依赖的图片、相关的论文

## result

- 记录一些分析结果。

## splits

- 用来存储训练集、验证集、测试集和推理回测集。
