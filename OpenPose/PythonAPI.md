# pyopenpose

## getPoseBodyPartMapping

getPoseBodyPartMapping(arg0: op::PoseModel) -> Dict[int, unicode]

## getPoseMapIndex

getPoseMapIndex(arg0: op::PoseModel) -> List[int]

## getPoseNumberBodyParts

getPoseNumberBodyParts(arg0: op::PoseModel) -> int

## getPosePartPairs

getPosePartPairs(arg0: op::PoseModel) -> List[int]

## get_gpu_number

get_gpu_number() -> int

获取可用GPU数量。

## get_images_on_directory

get_images_on_directory(arg0: unicode) -> List[unicode]

获取目录中的所有图片的相对路径。

# Datum

## Introduction

openpose整合各输入输出信息的数据结构。

```python
datum = op.Datum()
```

## cameraExtrinsics

## cameraIntrinsics

## cameraMatrix

## cvInputData

opencv读取的输入数据。

```python
    imageToProcess = cv2.imread(args[0].image_path)
    datum.cvInputData = imageToProcess
```

## cvOutputData

opencv所支持的输出结果数据。

```python
    cv2.imshow("result", datum.cvOutputData)
    cv2.waitKey(0)
```

## cvOutputData3D

## elementRendered

## faceHeatMaps

## faceKeypoints

ndarray

面部关键点数据。

## faceKeypoints3D

## faceRectangles

List[pyopenpose.Rectangle]

面部区域标识。

## frameNumber

## handHeatMaps

## handKeypoints

List[ndarray]

手部关键点数据，`handKeypoints[0]`为左手，`handKeypoints[1]`为右手。

## handKeypoints3D

## handRectangles

List[List[pyopenpose.Rectangle]]

手部区域标识

```python
    handRectangles = [
        # Left/Right hands person 0
        [
        op.Rectangle(320.035889, 377.675049, 69.300949, 69.300949),
        op.Rectangle(0., 0., 0., 0.),
        ],
        # Left/Right hands person 1
        [
        op.Rectangle(80.155792, 407.673492, 80.812706, 80.812706),
        op.Rectangle(46.449715, 404.559753, 98.898178, 98.898178),
        ],
        # Left/Right hands person 2
        [
        op.Rectangle(185.692673, 303.112244, 157.587555, 157.587555),
        op.Rectangle(88.984360, 268.866547, 117.818230, 117.818230),
        ]
    ]

    datum = op.Datum()
    datum.handRectangles = handRectangles
```

## id

## inputNetData

List[ndarray]

网络的输入数据。

## name

## netInputSizes

List[Point]

网络输入层的尺寸。

## netOutputSize

Point

网络输出层的尺寸。

## outputData

## poseCandidates

## poseHeatMaps

ndarray

## poseIds

## poseKeypoints

ndarray

身体关键点数据。

## poseKeypoints3D

## poseNetOutput

ndarray

```python
    datum = op.Datum()
    datum.cvInputData = imageToProcess
    datum.poseNetOutput = poseHeatMaps # 预先计算得到的datum.poseHeatMaps
    opWrapper.emplaceAndPop([datum])
```

## poseScores

## scaleInputToNetInputs

## scaleInputToOutput

## scaleNetToOutput

## subId

## subIdMax

# Point

## Introduction

## x

## y

# PoseModel

## name

# Rectangle

## Introduction

矩形表示结构。

重载的构造函数:

```python
__init__() -> None
```

```python
__init__(x: float, y: float, width: float, height: float) -> None
```

## <h2 id="Rectangle.height">height</h2>

矩形高度。

```python
face = datum.faceRectangles[0]
cv2.rectangle(image, (int(face.x), int(face.y)), (int(face.x + face.width), int(face.y + face.height)), (0, 0, 200))
```

## width

矩形宽度。另见 [Rectangle.height](#Rectangle.height)

## x

矩形横坐标。另见 [Rectangle.height](#Rectangle.height)

## y

矩形纵坐标。另见 [Rectangle.height](#Rectangle.height)

# WrapperPython

## Introduction

```python
opWrapper = op.WrapperPython()
```

获取openpose的python包装接口。

## configure

configure(self: openpose.pyopenpose.WrapperPython, arg0: dict) -> None

配置op的启动参数。

```python
params = dict()
params["model_folder"] = args.model_folder
params["face"] = args.face
opWrapper.configure(params)
```

## emplaceAndPop

emplaceAndPop(self: openpose.pyopenpose.WrapperPython, arg0: List[op::Datum]) -> None

读入数据并返回计算结果到datum。

```python
    datum = op.Datum()
    imageToProcess = cv2.imread(args[0].image_path)
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop([datum])

    cv2.imshow("OpenPose", datum.cvOutputData)
    cv2.waitKey(0)
```

## execute

execute(self: openpose.pyopenpose.WrapperPython) -> None     

## start

start(self: openpose.pyopenpose.WrapperPython) -> None

启动openpose wrapper。

```python
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
```

## stop

stop(self: openpose.pyopenpose.WrapperPython) -> None

终止openpose wrapper。

## <h2 id="WrapperPython.waitAndEmplace">waitAndEmplace</h2>

waitAndEmplace(self: openpose.pyopenpose.WrapperPython, arg0: List[op::Datum]) -> None

将数据读入openpose wrapper，异步操作。另见[waitAndPop](#WrapperPython.waitAndPop)。

```python
    opWrapper.waitAndEmplace([datum])
    print("Body keypoints: \n" + str(datum.poseKeypoints))
    time.sleep(5)
    print("Body keypoints: \n" + str(datum.poseKeypoints))
```

## <h2 id="WrapperPython.waitAndPop">waitAndPop</h2>

 waitAndPop(self: openpose.pyopenpose.WrapperPython, arg0: List[op::Datum]) -> bool

取出openpose wrapper的计算结果，异步操作。另见[waitAndEmplace](#WrapperPython.waitAndEmplace)。

```python
    datum.cvInputData = imageToProcess
    opWrapper.waitAndEmplace([datum])
    opWrapper.waitAndPop([datum])
```
