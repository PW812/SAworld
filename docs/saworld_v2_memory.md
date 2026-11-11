# SAWorld V2：随机当前帧与样本内残差 memory

本版恢复原 SAWorld 的当前帧训练方式：`DistributedGroupSampler` 随机采样，`filter_empty_gt=True`。先用原流程准备当前帧，若被过滤则按原 `_rand_another` 规则重采样；只有当前帧被接受后，才读取其历史。

## 每个样本独立构建历史

`SAWorldMemoryNuScenesDataset` 沿当前样本的 `prev` 链回溯同场景过去 K 帧，按最早到最近排列。缺少历史时保留 None 掩码，不重复当前帧充当有效历史，不跨场景、不读未来。相邻观测间隔超过 1 秒时停止回溯。

历史仅加载图像、相机参数、位姿和 CAN 状态，不进行 GT 检测/地图过滤，不加载深度标签或规划标签。历史的每个时间点仍有原 SAWorld 的相邻观测输入，用于计算该时间点的残差。提取历史区域时统一使用当前导航命令，不使用历史帧中由未来轨迹构造的命令。

历史图像在 DataLoader 中作为 CPU-only payload 保留。`SAWorldV2.build_history_memory` 每次只把一个历史时间点的有效 batch 子集搬到 GPU，使用当前模型权重在 `no_grad`、eval 模式下编码。历史不会更新 BN/SyncBN 统计；编码后恢复每个模块原来的 training 标记，再处理当前帧。

返回的 memory 为本次调用的局部字典，包含：

- `scene`、`residual`：`[B,K,N,C]`，场景 key 与残差 value；
- `points`：`[B,K,N,3]`，全局参考点；
- `timestamp`：`[B,K]`，实际时间戳（float64 秒）；
- `valid`：`[B,K]`，历史有效性。

默认 K=4、N=16、C=256，每个样本最多 64 个历史 token。参考点是 TokenLearner 加权 BEV 网格中心，不是物体检测中心。位姿与 RCSample 的 key-camera ego frame 一致，并补偿逆 BDA。

## 动作引导的读取

当前场景与当前残差组成查询，并读取全部 action token：

`Q = LN(S_t + R_t) + Attention(LN(S_t + R_t), A_t, A_t)`

将历史参考点变换到当前 BEV 坐标，以实际时间差编码历史年龄：

`C_t = Attention(Q + PE(P_t, 0), LN(S_memory) + PE(P_memory_to_current, age_seconds), R_memory)`

`R_future = ResidualDecoder(ActionMLN(R_t, prior_trajectory) + C_t)`

ActionDiT 槽位表示 DCT 运动系数，全部参与读取，不按导航命令选择其中一个槽位。后续继续使用 TokenFuser 生成未来 BEV，再以先验轨迹为参考点做可变形注意力细化。

历史 memory 全部 detach，读取层和当前 action 路径仍可反向传播。空 memory 采用零上下文且保留 DDP 所需梯度图。没有跨 batch 的读写、FIFO 或首帧预热依赖；训练和测试都从各自样本的实际历史构建 memory，测试结果不依赖此前测试过哪些样本。

`SAWorldHeadV2Sample` 和 `SAWorldV2` 独立实现 head/detector 主流程，不继承旧版本 head/detector。公共 BEVDepth4D、ActionDiT、TokenLearner、TokenFuser 和基础算子继续复用。

## 训练与对照

从 `SAWorld` 目录启动：

```bash
bash tools/dist_train.sh projects/configs/saworld/saworld_config_v2_memory.py 8
```

配置输出到 `work_dirs/saworld_v2_sample_memory`，避免覆盖旧流式实验。batch=16、lr=6e-4、12 epoch 等超参数保持当前配置设置。Loss 仍为 depth、prior planning、final planning、action flow 四项。

配置顶部 `memory_frames=4` 同时控制模型和 train/val/test 数据集的 K。做 K=0 消融时修改这个变量并重新加载配置。若使用命令行 `--cfg-options`，需同时覆盖 `model.pts_bbox_head.memory_frames` 和各数据集的 `memory_frames`，因为 Python 配置中的变量引用已在加载时求值。

旧流式版本单独保存在 `saworld_head_v2_streaming.py`，注册名仍为 `SAWorldHeadV2`，与 `SAWorld` detector 配合。旧 `work_dirs/saworld_v2_memory/saworld_config_v2_memory.py` 和旧 checkpoint 可以不修改，继续按原流式方式评测。

新样本内版本位于 `saworld_head_v2.py`，注册名为 `SAWorldHeadV2Sample`，与 `SAWorldV2` detector 和新 memory 数据集配合。若主动用新配置加载旧 v2 权重，则是在新历史协议下评测/微调，不能当作旧实验的原样验证。公平的新版本训练建议保持原 GeoBEV 初始化。

## 成本与边界

每个历史时间点都重新编码它自己的观测窗口，增加图像 I/O 和前向计算；这是保留随机 anchor 分布的代价。无梯度、逐时间点处理控制了中间激活显存，但 CPU 预取历史图像仍有额外内存开销。

历史参考点仅补偿自车位姿，不显式推断对象在遮挡期间的位置。历史特征可能过时，是否有效仍需验证集指标确认。旧 `SAWorldSceneSampler` 保留供旧配置参考，新配置不再使用它。

验证：

```bash
/xs-train-nas/workspace_cpw/conda/wam_cpw/bin/python -m unittest discover -s tests -p test_saworld_memory.py -v
```

StreamPETR 的 detached sparse feature 和全局参考点存储是设计参考：https://github.com/exiawsh/StreamPETR/blob/main/projects/mmdet3d_plugin/models/dense_heads/streampetr_head.py
