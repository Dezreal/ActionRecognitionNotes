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
epoch: 1	loss: 2.1987	acc: 0.078	val: 0.1739
epoch: 2	loss: 2.0310	acc: 0.276	val: 0.4348
epoch: 3	loss: 1.8460	acc: 0.359	val: 0.3478
epoch: 4	loss: 1.6731	acc: 0.344	val: 0.3913
epoch: 5	loss: 1.4437	acc: 0.510	val: 0.5217
epoch: 6	loss: 1.2534	acc: 0.568	val: 0.4783
epoch: 7	loss: 1.0844	acc: 0.646	val: 0.5652
epoch: 8	loss: 0.9571	acc: 0.656	val: 0.6522
epoch: 9	loss: 0.8714	acc: 0.661	val: 0.6522
epoch: 10	loss: 0.8501	acc: 0.677	val: 0.5217
epoch: 11	loss: 0.6810	acc: 0.755	val: 0.6957
epoch: 12	loss: 0.7056	acc: 0.729	val: 0.6522
epoch: 13	loss: 1.1009	acc: 0.609	val: 0.3043
epoch: 14	loss: 1.2163	acc: 0.552	val: 0.5217
epoch: 15	loss: 1.0215	acc: 0.625	val: 0.4783
epoch: 16	loss: 1.0115	acc: 0.620	val: 0.4783
epoch: 17	loss: 0.9302	acc: 0.646	val: 0.5217
epoch: 18	loss: 0.7324	acc: 0.714	val: 0.7391
epoch: 19	loss: 0.7280	acc: 0.714	val: 0.7391
epoch: 20	loss: 0.6889	acc: 0.729	val: 0.6957
epoch: 21	loss: 0.6339	acc: 0.724	val: 0.6957
epoch: 22	loss: 0.5918	acc: 0.781	val: 0.6957
epoch: 23	loss: 0.5793	acc: 0.781	val: 0.6522
epoch: 24	loss: 0.5630	acc: 0.781	val: 0.6522
epoch: 25	loss: 0.5405	acc: 0.802	val: 0.6522
epoch: 26	loss: 0.5176	acc: 0.792	val: 0.6522
epoch: 27	loss: 0.5033	acc: 0.786	val: 0.6087
epoch: 28	loss: 0.4886	acc: 0.797	val: 0.6087
epoch: 29	loss: 0.4635	acc: 0.839	val: 0.6087
epoch: 30	loss: 0.4436	acc: 0.833	val: 0.6087
epoch: 31	loss: 0.4311	acc: 0.828	val: 0.6087
epoch: 32	loss: 0.4185	acc: 0.828	val: 0.6087
epoch: 33	loss: 0.4019	acc: 0.844	val: 0.6087
epoch: 34	loss: 0.3783	acc: 0.849	val: 0.6087
epoch: 35	loss: 0.3513	acc: 0.870	val: 0.6087
epoch: 36	loss: 0.3253	acc: 0.891	val: 0.6087
epoch: 37	loss: 0.2988	acc: 0.891	val: 0.5652
epoch: 38	loss: 0.2687	acc: 0.906	val: 0.5652
epoch: 39	loss: 0.2407	acc: 0.922	val: 0.5652
epoch: 40	loss: 0.2194	acc: 0.938	val: 0.6087
epoch: 41	loss: 0.2959	acc: 0.891	val: 0.4783
epoch: 42	loss: 0.5021	acc: 0.828	val: 0.6957
epoch: 43	loss: 0.5324	acc: 0.802	val: 0.7391
epoch: 44	loss: 0.4738	acc: 0.854	val: 0.4783
epoch: 45	loss: 0.4180	acc: 0.833	val: 0.5217
epoch: 46	loss: 0.3078	acc: 0.901	val: 0.5217
epoch: 47	loss: 0.3085	acc: 0.885	val: 0.6957
epoch: 48	loss: 0.2601	acc: 0.885	val: 0.6957
epoch: 49	loss: 0.2511	acc: 0.906	val: 0.6522
epoch: 50	loss: 0.2170	acc: 0.927	val: 0.8261
epoch: 51	loss: 0.2166	acc: 0.932	val: 0.8261
epoch: 52	loss: 0.1866	acc: 0.953	val: 0.9130
epoch: 53	loss: 0.1521	acc: 0.969	val: 0.8261
epoch: 54	loss: 0.1438	acc: 0.943	val: 0.7391
epoch: 55	loss: 0.1142	acc: 0.974	val: 0.7391
epoch: 56	loss: 0.1031	acc: 0.984	val: 0.7391
epoch: 57	loss: 0.0933	acc: 0.984	val: 0.7391
epoch: 58	loss: 0.0787	acc: 0.990	val: 0.7391
epoch: 59	loss: 0.0609	acc: 0.995	val: 0.7826
epoch: 60	loss: 0.0558	acc: 0.995	val: 0.7391
epoch: 61	loss: 0.0507	acc: 0.995	val: 0.7391
epoch: 62	loss: 0.0431	acc: 1.000	val: 0.7826
epoch: 63	loss: 0.0387	acc: 1.000	val: 0.7826
epoch: 64	loss: 0.0327	acc: 1.000	val: 0.7826
epoch: 65	loss: 0.0282	acc: 1.000	val: 0.7826
epoch: 66	loss: 0.0263	acc: 1.000	val: 0.7391
epoch: 67	loss: 0.0223	acc: 1.000	val: 0.7391
epoch: 68	loss: 0.0183	acc: 1.000	val: 0.7826
epoch: 69	loss: 0.0160	acc: 1.000	val: 0.7826
epoch: 70	loss: 0.0143	acc: 1.000	val: 0.7391
epoch: 71	loss: 0.0121	acc: 1.000	val: 0.7391
epoch: 72	loss: 0.0102	acc: 1.000	val: 0.7391
epoch: 73	loss: 0.0091	acc: 1.000	val: 0.7391
epoch: 74	loss: 0.0086	acc: 1.000	val: 0.7826
epoch: 75	loss: 0.0077	acc: 1.000	val: 0.7826
epoch: 76	loss: 0.0067	acc: 1.000	val: 0.7826
epoch: 77	loss: 0.0059	acc: 1.000	val: 0.7826
epoch: 78	loss: 0.0054	acc: 1.000	val: 0.7826
epoch: 79	loss: 0.0051	acc: 1.000	val: 0.7826
epoch: 80	loss: 0.0047	acc: 1.000	val: 0.7826
epoch: 81	loss: 0.0043	acc: 1.000	val: 0.7826
epoch: 82	loss: 0.0038	acc: 1.000	val: 0.7826
epoch: 83	loss: 0.0034	acc: 1.000	val: 0.7826
epoch: 84	loss: 0.0032	acc: 1.000	val: 0.7826
epoch: 85	loss: 0.0030	acc: 1.000	val: 0.7826
epoch: 86	loss: 0.0028	acc: 1.000	val: 0.7826
epoch: 87	loss: 0.0027	acc: 1.000	val: 0.7826
epoch: 88	loss: 0.0025	acc: 1.000	val: 0.7391
epoch: 89	loss: 0.0023	acc: 1.000	val: 0.7391
epoch: 90	loss: 0.0021	acc: 1.000	val: 0.7391
epoch: 91	loss: 0.0020	acc: 1.000	val: 0.7391
epoch: 92	loss: 0.0019	acc: 1.000	val: 0.7391
epoch: 93	loss: 0.0018	acc: 1.000	val: 0.7391
epoch: 94	loss: 0.0017	acc: 1.000	val: 0.7391
epoch: 95	loss: 0.0017	acc: 1.000	val: 0.7391
epoch: 96	loss: 0.0016	acc: 1.000	val: 0.7391
epoch: 97	loss: 0.0015	acc: 1.000	val: 0.7826
epoch: 98	loss: 0.0015	acc: 1.000	val: 0.7826
epoch: 99	loss: 0.0014	acc: 1.000	val: 0.7826
epoch: 100	loss: 0.0014	acc: 1.000	val: 0.7826
testing
tensor(2) <- tensor(0)answer phone 接电话 <- wave 挥手
tensor(0) <- tensor(0)
tensor(1) <- tensor(1)
tensor(1) <- tensor(1)
tensor(7) <- tensor(2)read watch 看表 <- answer phone 接电话
tensor(3) <- tensor(2)clap 拍手 <- answer phone 接电话
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
tensor(8) <- tensor(7)bow 鞠躬 <- read watch 看表
tensor(7) <- tensor(7)
tensor(8) <- tensor(8)
tensor(8) <- tensor(8)
tensor(8) <- tensor(8)
tensor(8) <- tensor(8)
0.782608695652

Process finished with exit code 0

```

