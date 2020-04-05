# 关键点邻接图

参数 `frames_per_video` ：每一个视频序列所读取的帧数

## ST-GCN

参考经典的ST-GCN模型所使用graph构建邻接矩阵：

对于来自于同一个视频的各帧，不是构建 `frames_per_video`个图，而是将所有帧一起构建成同一个图，也就是说，一个图不是有25个节点，而是有`25 * frames_per_video`个节点，其中0 - 24是第一帧的各点，25 - 49是第二帧的各点，以此类推。

在邻接关系中，除了帧内的连接（如0 - 1, 5 - 6, 25 - 26, 30 - 31等）外，帧间对应点也建立连接关系，如0 - 25， 1 - 26， ...，25 - 50。以这种思路，将帧的时间序列定义进整个graph中。

## 邻接矩阵构建

以上述思路构建邻接矩阵代码实现（有向图）：

```python
    # pose_model = op.PoseModel_.BODY_25
    # number_parts = op.getPoseNumberBodyParts(pose_model)
    # part_pairs = op.getPosePartPairs(pose_model)
    part_pairs = [1, 8, 1, 2, 1, 5, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 9, 10, 10, 11, 8, 12, 12, 13, 13, 14, 1, 0,
                  0, 15, 15, 17, 0, 16, 16, 18, 2, 17, 5, 18, 14, 19, 19, 20, 14, 21, 11, 22, 22, 23, 11, 24]
    number_parts = 25

    edges = []
    # in_frame edges
    for i in range(0, frames_per_video):
        for j in range(0, len(part_pairs), 2):
            edges.append([part_pairs[j] + number_parts * i, part_pairs[j + 1] + number_parts * i])
    # frame_wise edges
    for i in range(0, frames_per_video - 1):
        for j in range(0, number_parts):
            edges.append([i * number_parts + j, (i + 1) * number_parts + j])

    edges = np.array(edges, dtype=np.uint16)
    adj = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                        shape=(number_parts * frames_per_video, number_parts * frames_per_video),
                        dtype=np.float32)
```

参考[pygcn](../GCN/PYGCN.md)中的方式对邻接矩阵进行对称化、归一化，并转化为PyTorch张量：

```python
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
```

（此时已成为无向图）