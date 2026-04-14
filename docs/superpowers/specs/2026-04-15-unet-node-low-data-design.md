# 基于预训练 U-Net Bottleneck-NODE 适配的低样本分割初步验证方案

## 1. 研究目标

本阶段的目标不是一次性搭建整篇论文的全部 baseline，而是以最小但严谨的实验闭环，优先验证如下核心假设：

在低样本分割场景下，固定预训练 U-Net 的 encoder，仅在 bottleneck 处引入 NODE 适配，并联合训练 decoder，可以改善特征表示并带来稳定的 Dice/IoU 提升。

本阶段以性能验证为主，几何分析为辅。是否“可行”的判断标准定义为：

Pretrained U-Net + NODE 相比 Pretrained U-Net，在相同训练与评估协议下，Dice/IoU 出现稳定提升。

这里的“稳定提升”优先指同一实验设定下重复运行结果方向一致，而不是单次偶然上涨。

## 2. 核心思路

本工作的理论出发点承接已有研究：此前已经观察到 NODE 对预训练分类模型的少样本微调具有改善特征几何结构的作用。基于这一经验，本工作将 U-Net 的前半部分视为一个来自分类预训练的特征提取器，认为 encoder 输出的 bottleneck 表征仍然保留了可进一步优化的几何结构。

因此，在 encoder 与 decoder 之间插入 NODE，不是为了替代 U-Net 原有结构，而是希望通过一个连续演化模块，对 encoder 输出特征进行几何重整，使其更适合后续 decoder 进行密集预测，从而提升最终分割表现。

本阶段优先选择“冻结 encoder，仅训练 decoder 与适配模块”的策略，原因如下：

1. 能够尽可能分离变量，避免性能提升被解释为 encoder 重新适配数据分布所致。
2. 更符合“参数高效微调”的思路，与前序分类任务中的 NODE 研究路线保持一致。
3. 在低样本场景下，冻结 encoder 能减轻过拟合风险，提高第一轮实验的结论清晰度。

## 3. 第一轮实验设定

第一轮实验采用 ISIC2018 作为验证数据集，并采用低样本训练设定进行初步验证。

固定设置如下：

1. 训练数据使用训练集的 10% 比例采样。
2. 验证使用固定全量验证集，不进行采样。
3. 使用固定随机种子 42 产生训练子集，保证不同方法看到完全相同的数据。
4. 所有实验组共享相同的训练轮数、优化器、学习率调度与数据增强策略。

选择 10% 作为第一轮档位，而不是极小 shot 数的原因是：

1. 相比 10-shot 或 20-shot，10% 具有更好的统计稳定性。
2. 样本量仍然较低，足以体现低样本微调难度。
3. 有利于快速得到可信的第一轮方向性结论。

## 4. 主实验对比组设计

第一轮主实验设置三组：

1. Group A: Pretrained U-Net, Frozen Encoder, Train Decoder Only
2. Group B: Pretrained U-Net, Frozen Encoder, Train Decoder + Vanilla Adapter
3. Group C: Pretrained U-Net, Frozen Encoder, Train Decoder + NODE Adapter

三组共享同一个预训练 encoder、同一个 decoder 主体，唯一差别位于 bottleneck 到 decoder 的连接位置。这样设计的目的是通过控制变量，严格比较不同 bottleneck 适配机制的效果。

三组的作用分别是：

1. A 组给出预训练 encoder 在当前任务上的基础能力。
2. B 组用于排除“仅仅增加一个可训练模块就能提升”的解释。
3. C 组用于验证 NODE 的连续演化机制是否优于普通适配。

第一轮不建议只做 A 对 C。虽然这样可以更快出结果，但一旦观察到性能提升，无法说明提升是否真正来自 NODE 的动力学特性，而不是额外参数或额外非线性变换。

## 5. 模型结构设计

整体模型结构统一为：

Input -> Frozen Pretrained Encoder -> Bottleneck Adapter -> Trainable Decoder -> Segmentation Output

其中：

1. Encoder 使用 ImageNet 预训练权重。
2. Encoder 全部冻结，不参与梯度更新。
3. Decoder 全部可训练。
4. Adapter 插入于 encoder bottleneck 与 decoder 输入之间。

A 组中，adapter 为恒等映射，不引入额外可训练瓶颈模块。

B 组中的普通适配器定义为：

1x1 Conv -> BatchNorm -> ReLU -> 1x1 Conv

这样做的优点是：

1. 表达能力足够强，能构成一个合理且有竞争力的对照组。
2. 计算代价小，适合 bottleneck 特征变换。
3. 方便与 NODE 内部函数的结构进行参数量级对齐。

C 组中的 NODE 适配器采用最小可运行形式：设 bottleneck 特征为 z0，通过 ODE 求解得到 zT，再送入 decoder。其动力函数 f(z, t) 使用与普通 adapter 尽量相似的卷积结构，例如：

1x1 Conv -> BatchNorm -> ReLU -> 1x1 Conv

这样设计可以最大程度保证：

1. B 组与 C 组参数规模接近。
2. 两者主要差异来自“普通离散变换”与“连续演化变换”的结构机制，而非底层算子差别太大。

第一轮 NODE 实现遵循“稳定优先、简单优先”的原则，不在首轮比较不同 solver。建议固定为单一求解策略，例如 Euler 或 RK4，并固定少量积分步数，先验证方向。

## 6. 训练协议

第一轮训练协议建议如下：

1. 损失函数采用 Dice Loss + BCE Loss。
2. 优化器采用 AdamW。
3. Encoder 学习率为 0，即完全冻结。
4. Decoder 与 Adapter/NODE 共享统一基础学习率。
5. 训练轮数先固定在 50 epochs 左右。
6. 采用基于验证集 Dice 的 early stopping 与 best checkpoint saving。
7. 每组实验训练预算严格一致。

选择 Dice + BCE 的原因在于 ISIC2018 为典型前景-背景二分类分割问题，该组合通常比单一 BCE 更稳，也比只用 Dice 更容易优化。

第一轮不建议引入过多训练技巧，例如多阶段学习率、复杂损失权重扫描或大量超参数搜索。当前目标是判断思路是否成立，而不是榨取最终最优性能。

## 7. 评估协议

第一轮主评估指标为：

1. Dice
2. IoU

同时可记录如下辅助指标，但不作为第一阶段核心判据：

1. Precision
2. Recall

评估时每组至少输出：

1. 最优 checkpoint
2. 训练损失曲线
3. 验证 Dice 曲线
4. 验证 IoU 曲线
5. 最终指标汇总表

第一轮的核心判断规则为：

1. 若 Group C 相比 Group A 的 Dice/IoU 稳定提升，则认为方向初步可行。
2. 若 Group C 进一步优于 Group B，则说明 NODE 的优势不仅来自“增加额外适配层”，而更可能来自其连续演化机制。

## 8. 几何分析的角色

本阶段几何分析不作为“是否可行”的主判据，仅作为辅助解释证据。

在第一轮实验中，建议仅对训练完成后的最佳模型执行 bottleneck 特征提取，并对 Group B 与 Group C 进行可视化对比，例如：

1. PCA/UMAP 投影
2. 类内紧致性观察
3. 类间可分性变化观察

这样做的目的不是拿几何图作为主要结论，而是在性能提升出现后，辅助说明 NODE 可能通过改善中间表示几何结构而带来收益。

如果第一轮性能没有明显提升，则不应由几何可视化替代性能结论。

## 9. 第一轮输出物

第一轮实验建议形成以下输出：

1. 一张主结果表，列出 A/B/C 三组在 ISIC2018 10% 设定下的 Dice 与 IoU。
2. 一组训练与验证曲线图，用于观察收敛速度和训练稳定性。
3. 一组 bottleneck 特征可视化图，作为机制辅助分析。
4. 一份简短结论，明确回答：冻结预训练 encoder、联合训练 decoder 与 NODE 是否能在低样本分割中带来稳定收益。

## 10. 后续可扩展方向

为了保证第一轮工作未来能平滑扩展为论文完整实验，系统设计上建议预留以下扩展接口：

1. adapter_type = none | conv | node
2. freeze_encoder = true | false
3. train_ratio = 0.01 | 0.05 | 0.1 | 0.2
4. seed = multiple
5. pretrained = true | false

在第一轮验证通过后，可进一步扩展到：

1. 多 seed 稳定性实验
2. 多比例低样本实验
3. 非预训练 U-Net 对照
4. 更强 baseline 比较
5. 更系统的几何分析与消融实验

## 11. 第一轮推荐精确定版

综合可行性、严谨性与实现成本，第一轮推荐采用如下固定版本：

1. 数据集：ISIC2018
2. 训练集：固定 10% 比例采样
3. 验证集：全量固定验证集
4. Encoder：ImageNet 预训练，全部冻结
5. Decoder：全部训练
6. 对比组：A decoder-only，B decoder+conv adapter，C decoder+NODE
7. Adapter 位置：encoder bottleneck 与 decoder 之间
8. Vanilla Adapter：1x1 Conv -> BN -> ReLU -> 1x1 Conv
9. NODE Function：与 Vanilla Adapter 尽量同构的卷积函数
10. 损失函数：Dice + BCE
11. 优化器：AdamW
12. 训练策略：固定 epoch，基于 val Dice early stopping，保存 best checkpoint
13. 主指标：Dice、IoU
14. 成功判据：C 相对 A 有稳定提升；若 C > B，则进一步支持 NODE 机制有效

## 12. 边界与后续计划

本技术路线文档只覆盖第一轮主实验设计，不包含以下内容：

1. 非预训练 U-Net 的对照实验
2. 同域场景下冻结 encoder 和 decoder、仅训练 NODE 的纯机制实验
3. 多个强分割 baseline 的系统横向比较
4. 正式论文阶段的全套消融矩阵

这些内容在第一轮主实验完成并确认方向可行后，再进入下一阶段规划。
