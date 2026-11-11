# SAWorld V2：动作引导的稀疏残差记忆

`SAWorldHeadV2` 在 `saworld_head_v2.py` 内独立实现层、前向和 loss，不导入或继承旧 head。共享的 ActionDiT、TokenLearner、TokenFuser 和基础算子继续使用公共模块。

本版只实现两项改动：保存历史场景残差，以及让当前动作检索历史、参与残差世界预测。配置基于 SAWorld 的 action 实验（每卡 batch 4，AdamW 2e-4，12 epoch），不启用前面的步序编码和 action-residual adapter。

## 记忆内容与生命周期

每个 batch slot 独立保存最近 `memory_frames=K` 帧，每帧有 `num_scenes=N` 个场景 key 和残差 value。默认 K=4、N=16，即最多读取 64 个历史 token。

记忆单元为 `(S_i, R_i, P_i, t_i)`：S 为当前帧经残差 TokenLearner 权重汇聚的场景特征，R 为同一权重下的相邻帧差分，P 为 BEV 网格中心的加权平均，t 为实际时间戳（秒）。P 是稀疏区域的粗略参考位置，不代表检测或跟踪到的物体中心。

参考 StreamPETR 的 `pre_update_memory` / `post_update_memory`：

- 读取前将历史全局参考点变换到当前 BEV 坐标，计算实际时间差。
- 先读过去 K 帧，再写入当前观测；写入的 feature 全部 detach，不缓存预测的未来或历史 autograd 图。
- FIFO 按帧截断，缓存不进入 state_dict/EMA checkpoint。
- scene_token、prev_idx 与 sample_idx 确认同一连续序列；重复、倒序、缺帧、超过 `memory_max_gap` 的间隔会清空对应 slot。切换 train/eval、epoch 或 batch 形状也会刷新。

位姿通过 `SAWorldMemoryMetadata` 从 `PrepareImageInputs` 的 key-camera ego pose 构建，与 `BEVDepth4D.prepare_inputs` 一致，并包含逆 BDA。参考点网格沿用 RCSample 的现有坐标排列，不改变基线的 BEV 表示。

## 动作引导的读取

首先由当前场景及残差查询全部 action token：

`Q = LN(S_t + R_t) + Attention(LN(S_t + R_t), A_t, A_t)`

随后加入空间/时间位置编码，使用历史场景作为 key，历史残差作为 value：

`C_t = Attention(Q + PE(P_t, 0), S_memory + PE(P_memory_to_current, age_seconds), R_memory)`

`R_future = ResidualDecoder(ActionMLN(R_t, prior_trajectory) + C_t)`

所有帧共享注意力和现有残差解码器。ActionDiT 的槽位表示 DCT 运动系数，因此全部用于读取，不能按导航命令索引其中一个槽位。

空记忆采用被严格屏蔽的零上下文，仍保留完整梯度图以支持 DDP；首帧与同权重 SAWorld 输出一致。之后继续使用 TokenFuser 生成未来 BEV，并以先验轨迹为参考点做可变形注意力细化。

Loss 仍为原有四项：depth、prior planning、final planning、action flow，没有新增监督。

## 训练与消融

从 `SAWorld` 目录启动：

```bash
bash tools/dist_train.sh projects/configs/saworld/saworld_config_v2_memory.py 8
```

模型输出目录：`work_dirs/saworld_v2_memory`。通过 `model.pts_bbox_head.memory_frames` 设置 K；设为 0 可在同一配置下做无记忆对照。

新配置使用 `SAWorldSceneSampler`：打乱场景顺序，保证同一 batch slot 按时间推进。每个 epoch 遍历全部样本，末尾补齐 DDP 所需数量。跨 stream 边界从空记忆开始。普通随机帧 sampler 会不断触发记忆刷新，不适合此版本训练。

配置还设置 `filter_empty_gt=False`：无检测框/地图线的帧仍保留规划与深度监督，避免父数据集随机替换样本而打断连续性。因此做 memory 消融时，应使用相同的新 sampler 和过滤设置，仅将 K 改为 0。

测试仍使用原连续 DistributedSampler，每个 rank/场景独立起始，边界前 K 帧逐渐填充缓存。缺失的历史不会用未来或其他场景填补。

单元验证：

```bash
/xs-train-nas/workspace_cpw/conda/wam_cpw/bin/python -m unittest discover -s tests -p test_saworld_memory.py -v
```

当前边界：记忆补偿自车位姿，不显式估计历史物体的运动；旧特征随模型更新可能过时。有限窗口和实际时间编码提供处理依据，但不保证所有历史都有益，需要训练后比较。

参考源码：https://github.com/exiawsh/StreamPETR/blob/main/projects/mmdet3d_plugin/models/dense_heads/streampetr_head.py
