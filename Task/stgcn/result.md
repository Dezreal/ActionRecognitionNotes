# 模型效果

```python
# x, l, tx, tl = load_features_for_stgcn("../data/features.txt", 8)
# labels = torch.tensor(l[:, 2] - 1, dtype=torch.long)
# tl = torch.tensor(tl[:, 2] - 1, dtype=torch.long)
# 
# # load graph
# graph_args = {"strategy": "spatial"}
# graph = Graph(**graph_args)
# 
# model = Model(in_channel, num_classes, graph, edge_importance_weighting=False)
# optimizer = optim.SGD(model.parameters(), lr=0.09, momentum=0.9)
# 
# model.train()
# for i in range(100):
```

```tex
/usr/bin/python2.7 /home/nya-chu/PycharmProjects/VideoActionClassifier/st_gcn/train.py
tensor(0) <- tensor(0)
tensor(2) <- tensor(0)answer phone 接电话 <- wave 挥手
tensor(1) <- tensor(1)
tensor(1) <- tensor(1)
tensor(7) <- tensor(2)read watch 看表 <- answer phone 接电话
tensor(7) <- tensor(2)read watch 看表 <- answer phone 接电话
tensor(3) <- tensor(3)
tensor(3) <- tensor(3)
tensor(3) <- tensor(3)
tensor(3) <- tensor(3)
tensor(7) <- tensor(3)read watch 看表 <- clap 拍手
tensor(4) <- tensor(4)
tensor(4) <- tensor(4)
tensor(5) <- tensor(5)
tensor(5) <- tensor(5)
tensor(6) <- tensor(6)
tensor(6) <- tensor(6)
tensor(7) <- tensor(7)
tensor(7) <- tensor(7)
tensor(8) <- tensor(8)
tensor(8) <- tensor(8)
tensor(8) <- tensor(8)
tensor(8) <- tensor(8)
82.609% (19/23)

Process finished with exit code 0
```

