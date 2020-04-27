# 滑动窗口

## 滑窗读取实现

https://github.com/Dezreal/VideoActionClassifier/blob/master/data/sliding.py

### 主要思路

以python生成器的方式迭代每个窗口的数据。

从视频第一帧开始依次经由OpenPose识别出关键点数据，逐帧计算，直至到达当前窗口的末尾帧，然后完成第一次返回。其后依次计算下一窗口所需要的、尚未处理的各帧，满足窗口要求后，进行下一个的返回。重复此过程，直至视频结束。所有计算过的数据都保存在List当中，每一帧不会被重复计算。缺点是，所有帧都要计算一次，即使某些帧从来不会被使用。

### 参数

- **path** (str) - 文件路径

- **width** (int) - 窗口大小
- **stride** (int, optional) - 滑动步幅
- **dilation** (int, optional) - 窗口内选取间隔
- **padding** (tuple, optional) - 左右零值填充

### 运行效果

滑动读取（`sliding(path, 8, stride=1, dilation=2, padding=(2, 2))`）并导入模型计算，视频实际内容是“挥手”。

```tex
/usr/bin/python2.7 /home/nya-chu/PycharmProjects/VideoActionClassifier/st_gcn/run.py
Starting OpenPose Python Wrapper...
Auto-detecting all available GPUs... Detected 1 GPU(s), using 1 of them starting at GPU 0.
['+', '0', '2', '4', '6', '8', '10', '12']
read watch 看表
['+', '1', '3', '5', '7', '9', '11', '13']
read watch 看表
['0', '2', '4', '6', '8', '10', '12', '14']
clap 拍手
['1', '3', '5', '7', '9', '11', '13', '15']
clap 拍手
['2', '4', '6', '8', '10', '12', '14', '16']
clap 拍手
['3', '5', '7', '9', '11', '13', '15', '17']
wave 挥手
['4', '6', '8', '10', '12', '14', '16', '18']
wave 挥手
['5', '7', '9', '11', '13', '15', '17', '19']
wave 挥手
['6', '8', '10', '12', '14', '16', '18', '20']
wave 挥手
['7', '9', '11', '13', '15', '17', '19', '21']
wave 挥手
['8', '10', '12', '14', '16', '18', '20', '22']
wave 挥手
['9', '11', '13', '15', '17', '19', '21', '23']
wave 挥手
['10', '12', '14', '16', '18', '20', '22', '24']
wave 挥手
['11', '13', '15', '17', '19', '21', '23', '25']
wave 挥手
['12', '14', '16', '18', '20', '22', '24', '26']
wave 挥手
['13', '15', '17', '19', '21', '23', '25', '+']
read watch 看表
['14', '16', '18', '20', '22', '24', '26', '+']
read watch 看表

Process finished with exit code 0
```



