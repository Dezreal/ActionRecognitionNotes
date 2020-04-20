

# ST-GCN代码思想

## Graph的建立

注：本模型构建的图的邻接矩阵尺寸为25*25，不再像[gcn](Task/gcn/graph.md)那样直接建立包含多帧的大图。

### 邻接矩阵的建立

```python
def get_hop_distance(num_node, edge, max_hop=1):
    """ 计算两个结点间的距离，距离大于`max_hop`的视为inf

    :param num_node: 结点总数
    :param edge: 边集
    :param max_hop: 最大距离
    :return: 距离矩阵
    """
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis
```

#### np.linalg.matrix_power(matrix, expo)

方阵阶乘，对于邻接矩阵：

2阶乘相当于将距离为2的两个点进行连接，3阶乘相当于将距离为3的两个点进行连接。

### 邻接矩阵预处理

```python
    def get_adjacency(self, strategy):
        """

        :param strategy:
        :return: 拆分了的多层的邻接矩阵，经normalize_digraph处理
        """
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        # 距离是'valid_hop'就是1
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)
```

归一化方式

```python
def normalize_digraph(A):
    '''
    Dn即 D^-1
    '''
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD
```

重点关注“spatial”抽样策略，或者说是节点划分方式。

center点为1号点（颈部）

```python
       elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
```

其实质就是将邻接矩阵拆分成多个子矩阵，子矩阵之和等于原矩阵。

## 网络的输入

整个网络的输入是一个(N = batch_size,C = 3通道即坐标维度,T = 帧数, V = 25关节数,M =  1人数)的tensor，所以在进行2维卷积(n,c,h,w)的时候需要将 N 与 M 合并起来形成(N * M, C, T,  V)换成这样的格式就可以与2维卷积完全类比起来。CNN中核的两维对应的是(h,w)，而st-gcn的核对应的是(T,V).

```python
...
def forward(self, x):

    # data normalization
    N, C, T, V, M = x.size()
    x = x.permute(0, 4, 3, 1, 2).contiguous()
    x = x.view(N * M, V * C, T)
    x = self.data_bn(x)
    x = x.view(N, M, V, C, T)
    x = x.permute(0, 1, 3, 4, 2).contiguous()
    x = x.view(N * M, C, T, V)
    ...
```

## Model

模型由3类层组成，其中层又有包含关系。每一个st_gcn又包含了residual模块。

```python
self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
self.st_gcn_networks = nn.ModuleList((
    st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
    st_gcn(64, 64, kernel_size, 1, **kwargs),
    st_gcn(64, 64, kernel_size, 1, **kwargs),
    st_gcn(64, 64, kernel_size, 1, **kwargs),
    st_gcn(64, 128, kernel_size, 2, **kwargs),
    st_gcn(128, 128, kernel_size, 1, **kwargs),
    st_gcn(128, 128, kernel_size, 1, **kwargs),
    st_gcn(128, 256, kernel_size, 2, **kwargs),
    st_gcn(256, 256, kernel_size, 1, **kwargs),
    st_gcn(256, 256, kernel_size, 1, **kwargs),
))
# initialize parameters for edge importance weighting
if edge_importance_weighting:
    self.edge_importance = nn.ParameterList([
        nn.Parameter(torch.ones(self.A.size()))
        for i in self.st_gcn_networks
    ])
else:
    self.edge_importance = [1] * len(self.st_gcn_networks)
# fcn for prediction
self.fcn = nn.Conv2d(256, num_class, kernel_size=1)
```

每一层st-gcn的搭建：

```python
self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])
# temporal
self.tcn = nn.Sequential(
    nn.BatchNorm2d(out_channels),
    nn.ReLU(inplace=True),
    nn.Conv2d(
        out_channels,
        out_channels,
        (kernel_size[0], 1),
        (stride, 1),
        padding,
    ),
    nn.BatchNorm2d(out_channels),
    nn.Dropout(dropout, inplace=True),
)

if not residual:
    self.residual = lambda x: 0

elif (in_channels == out_channels) and (stride == 1):
    self.residual = lambda x: x

else:
    self.residual = nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=(stride, 1)),
        nn.BatchNorm2d(out_channels),
    )

self.relu = nn.ReLU(inplace=True)
```

每一个st-gcn层都用residual模块来改进。可以在源码中看出来当通道数要增加时，使用1x1conv来进行通道的翻倍，另外使用`stride = 2`来完成pool的效果使得长宽减半。

```python
self.gcn = ConvTemporalGraphical(in_channels, out_channels,kernel_size[1]) #使用卷积核的第二维即 3 组

self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1), #使用卷积核的第一维即 9 帧
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )
```

## GCN与TCN模块

### GCN

```python
self.gcn = ConvTemporalGraphical(in_channels, out_channels,kernel_size[1])
```

```python
class ConvTemporalGraphical(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size

        x = self.conv(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))

        return x.contiguous(), A
```

### TCN

在GCN后面紧跟着就是TCN的模块，该模块让网络在时域中进行特征的提取，类似与LSTM，GCN的输出是一个(n,c,t,w)的blob，在TCN中可以简单的理解为和CNN的输入格式一样。

```python
self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels, # 不改变chanl值
                (kernel_size[0], 1),
                (stride, 1), # stride可以控制t域的缩小，可当做poolling操作
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )
```

