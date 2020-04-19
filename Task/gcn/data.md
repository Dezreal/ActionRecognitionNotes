# 数据预处理

## 数据来源

[Florence 3D actions dataset](https://www.micc.unifi.it/resources/datasets/florence-3d-actions-dataset/)

2012年佛罗伦萨大学用Kinect相机捕获，包括挥手、喝水等9种动作，共有10名演员，总共包含215个视频序列。

下载的数据集中附有作者已处理完成的特征数据文本文件，但是本任务中并没有使用。

## 特征提取

本任务使用OpenPose的python API提取所需要的关节特征。

使用OpenPose的BODY_25模型，其中：

关节数：25

关节id:

```
{ 
 {0,  "Nose"},
 {1,  "Neck"},
 {2,  "RShoulder"},
 {3,  "RElbow"},
 {4,  "RWrist"},
 {5,  "LShoulder"},
 {6,  "LElbow"},
 {7,  "LWrist"},
 {8,  "MidHip"},
 {9,  "RHip"},
 {10, "RKnee"},
 {11, "RAnkle"},
 {12, "LHip"},
 {13, "LKnee"},
 {14, "LAnkle"},
 {15, "REye"},
 {16, "LEye"},
 {17, "REar"},
 {18, "LEar"},
 {19, "LBigToe"},
 {20, "LSmallToe"},
 {21, "LHeel"},
 {22, "RBigToe"},
 {23, "RSmallToe"},
 {24, "RHeel"}
 }
```

连接关系：

```
[1, 8, 1, 2, 1, 5, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 9, 10, 10, 11, 8, 12, 12, 13, 13, 14, 1, 0, 0, 15, 15, 17, 0, 16, 16, 18, 2, 17, 5, 18, 14, 19, 19, 20, 14, 21, 11, 22, 22, 23, 11, 24]
```

### 主要人物选取

许多视频背景中存在背对镜头坐在桌前工作的无关人员，需要将这些特征删去，考虑其特点，采用比较其所占画面比例的方法进行筛选，将占比小的人员筛除。这一方法在本任务中准确有效。

识别四方向位置：

```python
        widths = one[:, 1]
        widths = widths.ravel()[np.flatnonzero(widths)]
        left = widths.min()
        right = widths.max()
        heights = one[:, 0]
        heights = heights.ravel()[np.flatnonzero(heights)]
        top = heights.min()
        bottom = heights.max()
```

### 生成特征文件

对215个视频都逐帧进行关键点识别，将主要人物的特征数据记录在一个特征文件`features.txt`里，文件每一行对应一帧，对于其每行：

[0]为action id

[1]为actor id

[2]为category id，也就是分类标签

[3:6]为一号关节的特征数据，其后以此类推

[-1]为当前帧数

最终共提取4016帧的关键点特征，形成数据文件。

## 数据加载

参数 `frames_per_video` ：每一个视频序列要读取的帧数

首先根据标签信息划分每行属于哪一视频序列，对同一个视频序列的各行数据，各自间隔均匀地选取`frames_per_video`帧作为数据。

```python
    # the index of frames we need
    index = []
    for i, start in enumerate(frames_0th):
        if i == num_sequences - 1:
            end = data.shape[0] - 1
        else:
            end = frames_0th[i + 1] - 1
        index.extend(np.linspace(start, end, frames_per_video, dtype=np.uint16).tolist())

    index = [i for item in index for i in item]
    data = data[index, :]
```

其次根据actor标签进行排序，以便于后续根据actor id划分训练集和测试集，同时以action、category和当前帧标签作为次要排序条件，以保证帧顺序不因排序而打乱。

```python
    # sort by (actor > action > category > frame)
    actor = data[:, 1]
    frame = data[:, -1]
    action = data[:, 0]
    category = data[:, 2]
    index = np.lexsort((frame, category, action, actor))
    data = data[index]
```

然后完成对数据进行拆分，分出数据与标签；划分训练集与测试集；在视频序列间打乱顺序三个操作。

数据的尺寸`shape = (n_videos, 25 * frames_per_video, 3)`，其中，

dim 0是不同视频序列；

dim 1中：

[0:25]是第1帧1号关节到25号关节的特征

[25:50]是第2帧1号关节到25号关节的特征，以此类推；

dim 2分别是关键点的x，y，置信度。

最后对`dim = 1`进行归一化处理，完成数据预处理。

归一化函数（dim = 1）：

```python
def normalize_on_dim(arr, dim):
    max = np.max(arr, axis=dim, keepdims=True)
    min = np.min(arr, axis=dim, keepdims=True)
    _range = max - min
    return (arr - min) / _range
```

