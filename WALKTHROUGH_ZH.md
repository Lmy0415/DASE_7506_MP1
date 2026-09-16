# MP1 中文理解与答辩笔记

这份笔记不是报告的替代品，而是帮助你真正理解自己提交了什么、为什么
这样比较、以及老师可能追问什么。

## 1. 任务到底是什么

模型看到当前及之前 token，预测下一个 token：

$$p(x_{t+1}\mid x_{\le t}).$$

训练使用 cross-entropy。对整段文本把正确 token 的负对数概率相加，再把
自然对数单位换成 bit，并除以原始 UTF-8 byte 数，就是 BPB：

$$
\mathrm{BPB}=\frac{\sum_t-\ln p(x_{t+1}\mid x_{\le t})}
{B\ln 2}.
$$

所以 lec 1 中的 next-token cross-entropy 和 MP1 的 BPB 是同一件事的两种
归一化：训练降低平均 token loss，评测把整个 split 的总 loss 按 byte 报告。
正确 token 概率越高，$-\log p$ 越小，BPB 越低。Top-1 accuracy 不够，因为
它看不出模型给正确 token 0.51 还是 0.99 的差别。

## 2. Baseline（对应 lec 2 的 Transformer）

Baseline 是 4 个 pre-norm Transformer blocks，width 128、4 heads、context
256，共 1,088,256 个参数。每个 block 的顺序是：

```text
x
 ├─ LayerNorm → one QKV projection → causal multi-head attention → output projection ─┐
 └─────────────────────────────────────────────────────────────────────────────────────+→ x'

x'
 ├─ LayerNorm → Linear(128,512) → GELU → Linear(512,128) ─┐
 └──────────────────────────────────────────────────────────+→ block output
```

Q、K、V 合在一个 `Linear(width, 3*width)` 里只是一次高效矩阵乘法，之后再
切开；数学上仍是三组投影。4 heads 时每个 head 是 $128/4=32$ 维。causal
mask 必须在 softmax 前把未来位置设为 $-\infty$，这样指数为 0，未来 token
得到零权重，不需要 softmax 后再次归一化。

Residual 的意义不是“把 attention 重复很多次”，而是让每层学习对当前
representation 的修正，并为梯度提供稳定通道。堆叠更多 blocks 会进行多轮
attention + MLP 推理；一个 gate 只是在某层缩放一次 residual，两者表达能力
和计算代价都不同。本提交最终没有使用 gate。

## 3. 我们做的三项改变

### 3.1 更合适的模型容量

最终神经模型是 width 320、8 blocks、10 heads，共 10,601,600 参数。每个
head 仍是 32 维，因此改变的是总通道容量和层数，不是单 head 尺度。

320x8 不是拍脑袋决定的。探索阶段在约一千万参数、600 updates、同一 seed
下比较了三种 shape：

| Shape | 参数 | Validation BPB |
|---|---:|---:|
| 256x12 | 10,067,456 | 2.137945 |
| 288x10 | 10,654,848 | 2.107498 |
| 320x8 | 10,601,600 | **2.076717** |

在这个有限 search space 和短训练预算中，较宽较浅的 320x8 最好。不能把
它写成普遍定律，只能说它是这三个候选中的验证集赢家。

正式 seed 7506 的同 targets 证据更强：matched 128x4 是 1.997825 BPB，
320x8 是 1.780348，差值 **-0.217477 BPB**。两者都处理 9,830,400 个
targets，batch 顺序和 LR trajectory 相同，因此这个差值可以主要归因于
architecture/capacity，而不是更长训练。

### 3.2 更慢的 cosine 退火和更多训练

Baseline 原本在 1,200 steps 内从 peak LR 降到接近最低值；正式 B0 在最后
一步的 LR 约为 $1.0\times10^{-4}$。B1 和 P0 把 cosine horizon 设为 4,800，
虽然公平比较仍停在 step 1,200，但此时 LR 仍为 $8.684\times10^{-4}$。

只改这个 horizon，baseline validation BPB 从 2.075891 降到 1.997825，
改善 **0.078066**。这说明“慢退火”本身就是重要贡献，所以报告必须把它
和“大模型贡献”分开。

P0 最终继续到事先固定的 step 3,000，共 24,576,000 targets。探索结果显示
cosine 在 step 3,000 达到 1.644901，之后没有继续改善；WSD 在后期追回，
但其最好观察值仍未超过 cosine step 3,000。因为这是 10% mini project，
正式方案没有继续搜索 WSD split，而是把最有证据、成本较低的设置锁定复现。
正式 seed 7506 的 P0 在 step 1,200 是 1.780348 BPB，继续到 step 3,000
且 cache 关闭时是 1.634624 BPB，因此额外训练的独立贡献是
**-0.145724 BPB**。这里只能说这条固定 trajectory 在这个 seed 上有效，不能
据此宣称对所有初始化都稳定。

### 3.3 Strictly causal continuous cache

神经模型给出 $p_{NN}$。cache 查看当前 256-token window 中与当前 hidden
state 相似的较早 hidden states，并统计那些位置后面已经观察到的 successor
token，得到 $p_{cache}$：

$$p=(1-0.065)p_{NN}+0.065p_{cache}.$$

三个参数的含义：

- `window=255`：256-token window 内最多只有 255 个更早位置；不是 265。
- `theta=13`：放大 cosine similarity 后再 softmax，控制 retrieval 集中程度。
- `lambda=0.065`：只让 cache 占 6.5%，神经模型仍占 93.5%。

因果性关键在 index：预测位置 $t$ 只允许 memory $m<t$；memory 的 value 是
$x_{m+1}$。因为 $m+1\le t$，这个 value 已经在输入 prefix 中，不是要预测的
$x_{t+1}$。cache 每次 `predict_log_probs` 都重新建立，绝不跨 window 保存，
所以没有 KV cache 式“把上一个独立窗口带进来”的作弊。

正式 cache-off 是 1.634624 BPB，cache-on 是 1.598754 BPB；两者来自完全
相同的 step-3,000 neural weights，所以 **-0.035870 BPB** 可以归因于 cache
开关。这是整个项目最干净的 mechanism ablation。

## 4. 为什么需要四个 delta

最终 BPB 改善不是全都来自“大模型”，而是四部分：

1. `B1 - B0`：只改变 LR horizon，得到 schedule effect。
2. `P0@1200 - B1`：同 targets、同 trajectory，得到 architecture effect。
3. `P0@3000 off - P0@1200`：同一模型多训练，得到 additional-training effect。
4. `P0@3000 on - P0@3000 off`：同一份权重只开关 cache，得到 cache effect。

四项相加等于最终 validation BPB 相对 B0 的总变化。这样才不会把慢退火、
更多 targets 或 cache 的收益错误地全部记在 model scaling 名下。

| Effect | BPB delta（负数更好） |
|---|---:|
| Schedule horizon | -0.078066 |
| Architecture | -0.217477 |
| More training | -0.145724 |
| Cache | -0.035870 |
| Total | -0.477137 |

四项使用未四舍五入的数相加，最终 validation 从 2.075891 降到 1.598754，
总降幅是 **0.477137 BPB，或 22.9847%**。这个百分比是相对 B0 的 BPB 降幅，
不是 accuracy 提升。

## 5. 合规性你要能自己解释

- 只用课程 train text，从随机初始化训练；没有 pretrained weights 或外部语料。
- tokenizer、data、`common.py` 和 `evaluate.py` 不改。
- development copy 物理删除 test；训练和 validation scorer 没有 test API。
- 正式比较统一 seed 7506；不用旧 seed 17 冒充正式证据。
- cache 不看 future token、不跨 sample/window 保存 state。
- 最终评测是 CPU FP32；目标是时间不超过 paired baseline 5x、RAM 小于 4 GiB、
  inference assets 小于 64 MiB。
- 一个 seed 只能支持“本次 paired run 的结果”，不能声称 statistically
  significant、robust 或对所有 seed 都稳定。

正式资源测量已经通过：同一 validation score loop 中，B0 用 8.209438 秒，
P0 cache-on 用 34.338785 秒，比例 **4.182842x**；P0 peak RSS 是
**1.517380 GiB**，uncompressed inference assets 是 **40.478294 MiB**。
三项分别低于 5x、4 GiB 和 64 MiB 限制。外部端到端 elapsed time 另测得 B0
16.55 秒、P0 42.88 秒（2.590937x）；它和 scorer 内部 timing 不是同一口径，
不能混着算。

成本也要按口径说明：B0+B1 合计 765.958 process seconds；P0 训练本体
4,775.762 秒、整个 P0 process 4,843.256 秒。三个正式 pre-test resource
scoring 命令外部 elapsed 合计 100.55 秒，freeze 与 contract tests 约
12.874 秒。冻结 commit `30eb20d9d1f36bb790615c3753fa2c6e8ba53b5f`
之后只运行了一次新的正式 final test，得到 **1.6160026140038766 BPB**；
score loop 是 41.520049 秒，外部 elapsed 是 50.96 秒，peak RSS 是
1.550232 GiB。这个结果只用于最终汇报，没有再据此调整模型。探索阶段可证实
的成本下界是 40,065.386 job-seconds；cache
grid 和无 artifact 的失败任务因没有时间记录，只能标记 unknown，不能记成 0。

历史上曾经有 5 次探索性 test evaluation。重新解压 ZIP 不能让研究者忘掉
已经见过的分数，所以必须披露。我们能做到的是：之后预注册正式设置、正式
开发不读取 test、冻结 checkpoint/hash 后只运行一次新的正式 test，并且不按
这个结果改方法。

## 6. 最后你应该能用一分钟讲清楚

> 我先复现 128x4 baseline，再增加一个相同 targets、相同 batch 和相同慢
> cosine trajectory 的 matched baseline。然后用 320x8 在 step 1,200 做
> same-target architecture control，并按预注册继续到 step 3,000。最终在同一
> 份神经权重上开关 strictly causal within-window continuous cache，作为关键
> mechanism ablation。这样最终 BPB 可以拆成 schedule、architecture、额外
> training 和 cache 四部分，而不是把所有提升都归因于更大的模型。正式 test
> 只在代码、cache 参数和 checkpoint SHA-256 冻结后运行一次。

你还应能回答：为什么 320x8 是有限搜索结果、为什么 LR horizon 和 actual
stop 分开、cache 的
value 为什么没有泄漏 target、资源计时为何有 score-loop 与 external elapsed
两种口径，以及 single seed 和历史 test exposure 限制了哪些结论。
