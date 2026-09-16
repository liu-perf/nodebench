---
layout: default
title: nodebench
---

# nodebench

**5 分钟测完一整台多卡 GPU 节点，并给出一份说清楚数字含义的报告。**

nodebench 覆盖 FLOPS、显存带宽、PCIe、NCCL 四项测量，每次运行前先跑一段空闲对照组当噪声基线，关键指标用 PyTorch 和原生 C++ 工具链（cuBLAS/CUTLASS、BabelStream、nvbandwidth、nccl-tests）双实现交叉验证，报告末尾还会列出哪些模块没跑、为什么没跑。无需编译、无需数据集、无需 root，有 PyTorch 和 NVIDIA 驱动就能跑。

项目主页与安装使用说明见 [README](../README.md)（仓库地址：`https://github.com/liu-perf/nodebench`）。

📊 **[七部曲关键数字 Dashboard](dashboard.html)** — P2P 比值 0.22→0.72、benchdoctor 假阳性 13→0、tracedoctor 假阳性 73x→1.5x、跨工具吻合 0.1%、fitdoctor KV cache 偏大 4.0x、telemetrydoctor 同一利用率读数下算力差 978x、servedoctor 闭环压测尾延迟短 16.5x，一页看完。自包含，无 JS 库、无 CDN，断网能开。

---

## 技术笔记

这几篇是写代码过程中踩坑之后整理的方法论，仓库的 `docs/` 目录下都能找到：

- [`p2p-explained.md`](p2p-explained.md) — 为什么四种标准工具都判定"没有 P2P"，节点却在用一条它们都看不见的快速通路；以及 nodebench 如何用 all-reduce busbw / H2D 带宽的比值来判断真实行为而不是硬件能力。
- [`nvml-pitfalls.md`](nvml-pitfalls.md) — NVML 里四个名字和含义对不上的计数器：PCIe 吞吐是 20ms 窗口均值不是字节数，`utilization.gpu` 不是 SM 占用率，采样间隔并不等于你设的间隔，throttle 是一个位掩码而不是布尔值。
- [`compat-matrix.md`](compat-matrix.md) — CUDA 架构、torch 版本、编译期 arch list 三层检查，其中第三个（`torch.cuda.get_arch_list()`）是最常被忘记、也最容易在新卡上炸掉的一环。
- [`methodology.md`](methodology.md) — nodebench 的十条方法论：空闲对照组、双实现交叉验证、best/median 双报告、环形算法理论核对、busbw 而非 algbw、按活跃卡集而非时间窗分箱、同 NUMA/跨 NUMA 对照、全链路 provenance 等。

---

## 跑起来长什么样

`doctor` / `report` / `run` / `pytest` 四条命令。

`report` 和 `pytest` 两条命令在任何一台笔记本上都能跑，不需要 GPU——31 个测试 0.11 秒跑完，一次都没碰显卡。`run` 没有给出终端记录，因为那是你自己机器的数字，编一份看起来合理的贴进文档，恰恰就是这一整套工具想抓的那类错误。

---

## 延伸阅读 / Writing

以下几篇是围绕 nodebench 开发过程写的文章，链接待发布后补充：

- 四个工具都说不支持 P2P，但它一直在用 — 链接待发布后补充
- 四个 NVML 计数器，名字和含义对不上 — 链接待发布后补充
- sm_120 × CUDA × torch：三个检查，第三个所有人都忘 — 链接待发布后补充
- 同样是 8 张卡，插在同一个 CPU 还是拆成两组，通信带宽差了三分之一 — 链接待发布后补充
- 一个 benchmark 数字要满足什么条件才可信 — 链接待发布后补充
- nodebench：一套面向 AI GPU 的节点级 benchmark 框架 — 链接待发布后补充

---

## 姊妹项目

nodebench 是"测量 → 静态找坑 → 动态找坑 → 时间维度找坑 → 该是多少"七部曲的第一部：

- [benchdoctor](https://github.com/liu-perf/benchdoctor) — 静态找坑：在代码和配置层面分析 benchmark 是否存在方法论问题，属于七部曲的第二部。它的每一条规则都是写 nodebench 时手工踩过一次坑之后才总结出来的。
- [tracedoctor](https://github.com/liu-perf/tracedoctor) — 动态找坑：通过真实运行时的 trace 分析定位性能问题，属于七部曲的第三部。它的 TD004 规则要靠 nodebench 的 PCIe benchmark 提供参照数（同一块卡上两个工具各自测出 26.46-26.48 和 26.48 GB/s，差 0.1% 以内）。
- [regressiondoctor](https://github.com/liu-perf/regressiondoctor) — 时间维度找坑：读两份 nodebench 的 `results.json`，判断"这次比上次慢"是真退化还是测量噪声，属于七部曲的第四部。判定的杠按每个指标自己记录的 `cv_pct` 来定，而不是一刀切的固定百分比。
- [fitdoctor](https://github.com/liu-perf/fitdoctor) — 容量与上限，属于七部曲的第五部，也是唯一换了方向的一部：前四部问"测出来的数对不对"，它问"这个数本来该是多少"（显存装不装得下、瓶颈在带宽还是算力）。它自己写了一张按架构比值推导上限的表（`ceilings.py` 的 `ARCH_RATIO_VS_BF16`），四张真卡上 tf32/fp16/fp8 稳在 1.4% 以内，然后 fp4 完全不成立（同代卡的 fp8→fp4 是 ×2.87 / ×1.37 / ×1.98），于是对 fp4 直接拒绝推导。**这里原来写的是「它缩小了 `methodology.md` 里那条按架构固定比值交叉验证的作用域」——nodebench 没有那条方法论**（第 2 条是同 dtype 的双实现交叉验证），指过去的东西不存在，这句已撤回。
- [telemetrydoctor](https://github.com/liu-perf/telemetrydoctor) — 监控口径审计，七部曲的第六部。给每一列监控数据建「聚合契约」：`power` 可以积分成焦耳，PCIe 速率乘以时长会错 69 倍，SM 百分比不能跨卡相加。它同时**缩小了 `nvml-pitfalls.md` 里关于 `utilization.gpu` 的说法**——那一列是时间口径不是强度口径，本机实测两个算力差 978 倍的负载报出的利用率只差 1.7 个点，所以它只用忙卡的**集合**、从不用忙碌的**程度**。
- [servedoctor](https://github.com/liu-perf/servedoctor) — 压测方法审计，七部曲的第七部，也是最后一部。前六部审视的都是被测系统，它把镜头转过来对着**测量装置本身**：这份压测报告测的是服务，还是压测器。纯标准库实测，同一台服务器、同一个配置速率，闭环压测报出的 q99 比开环短 **16.5 倍**，而两者最大值只差 1.8 倍——**这是七个里唯一一个不需要 GPU 就能自己复现的**。
