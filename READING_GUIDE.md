# CS 185/285 Spring 2026 文献阅读路线



## 0. 最低前置

- [Sutton & Barto, *Reinforcement Learning: An Introduction*](http://incompleteideas.net/book/the-book-2nd.html)：先读第 3、4 章，弄清 MDP、return、value function、Bellman equation 和 policy iteration。本地已有 [教材 PDF](SuttonBartoIPRLBook2ndEd.pdf)，不必一开始通读全书。
- [Sutton, *The Bitter Lesson*](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)：对应 `L01 p.30`。这是一篇短文，用来理解课程为什么强调“学习 + 搜索”，不是算法论文。

## 第一阶段：模仿学习

### 1. DAgger：先理解分布偏移

- 文献：[Ross et al., *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*](https://proceedings.mlr.press/v15/ross11a.html)
- Slides：`L02 p.31–37`，`L21 p.8–9`。
- 核心：行为克隆在专家状态分布上训练，却在自己的状态分布上运行，错误会沿轨迹累积；DAgger 反复收集学习者访问到的状态并请专家标注，从根源上修正训练分布。
- 第一遍重点：引言、算法 1、误差随 horizon 增长的结论；先跳过完整 no-regret 证明。
- 读完检查：能否解释为什么普通监督学习误差小，不等于闭环控制表现好。

### 2. Causal Confusion：更多输入也可能更差

- 文献：[de Haan et al., *Causal Confusion in Imitation Learning*](https://proceedings.neurips.cc/paper_files/paper/2019/hash/947018640bf36a2bb609d3557a285329-Abstract.html)
- Slides：`L03 p.8`。
- 核心：模型可能把动作造成的结果误当成动作的原因；这种伪相关在训练集上很好用，部署后却会失效。
- 第一遍重点：刹车灯例子、causal misidentification 的定义、干预为何能识别正确因果关系。
- 读完检查：能否区分 distribution shift 和 causal confusion，它们相关但不是同一问题。

### HW1 关联：生成策略与机器人模仿学习

- 先读 [Peter Roelants, *Flow Matching* 教程](https://peterroelants.github.io/posts/flow_matching_intro/)（`L03 p.14`、`L21 p.12` 的插图来源），对照 HW1 的直线路径、速度场回归和 Euler 采样。HW1 实现的是 flow matching，不能把它与扩散模型训练公式直接混用。
- [Chi et al., *Diffusion Policy*](https://diffusion-policy.cs.columbia.edu/)（`L03 p.16–17`、`HW1 p.1,4`）：用条件扩散过程表示多峰、高维动作分布，并通过 action chunking 做闭环控制。HW1 将它列作背景文献；先看策略表达与多峰动作，不必复现整篇的视觉架构。课件写 2023/2024，HW1 引 2025 期刊版，按同一工作合并。
- `GNM: A General Navigation Model to Drive Any Robot`（`L03 p.31`）：关注大规模、跨机器人目标条件行为克隆；适合对机器人导航感兴趣时再读。

## 第二阶段：深度强化学习主干

### 先补策略梯度与 GAE（HW2 主线）

- 先读 `L05–06` 和 `D03`，弄清 REINFORCE、reward-to-go、baseline 与 TD residual；再读 [Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation*](https://arxiv.org/abs/1506.02438)。
- 明确出处：`L06 p.24`、`L21 p.32`、`D03 p.8`、`HW2 p.3`。
- 第一遍重点：`δ_t = r_t + γV(s_{t+1}) − V(s_t)`，以及 `Â_t = Σ_l (γλ)^l δ_{t+l}`；区分价值函数训练目标和策略梯度里的 advantage。
- 读完检查：能解释 λ 控制的偏差/方差折中，以及 episode 结束处的 bootstrap/mask 如何影响递推。


### 3. DQN：价值学习如何进入深度网络

- 文献：[Mnih et al., *Human-level Control through Deep Reinforcement Learning*](https://www.nature.com/articles/nature14236)
- Slides：`L04 p.40`，`L08 p.7`；2013 预印本和 2015 Nature 版在 slides 中都出现，主线读 2015 版即可。
- 核心：experience replay 打破样本相关性，target network 减慢 bootstrap 目标变化，两者让卷积网络上的 Q-learning 更稳定。
- 第一遍重点：算法流程、replay buffer、target network；先不纠结 Atari 预处理和所有实验表格。
- 读完检查：能否指出“数据分布、目标值、策略”三者为什么都在变化。
- 接着读 [van Hasselt et al., *Deep Reinforcement Learning with Double Q-learning*](https://arxiv.org/abs/1509.06461)（`D04 p.1–3`，HW3 的相关实现）：online Q 选动作，target Q 评价。不要把 Double DQN 和 SAC/TD3 的两个 critic 取最小值当成同一种更新。

### 4. TRPO → PPO：限制每次策略更新的幅度

- 文献：[Schulman et al., *Trust Region Policy Optimization*](https://proceedings.mlr.press/v37/schulman15.html) → [Schulman et al., *Proximal Policy Optimization Algorithms*](https://arxiv.org/abs/1707.06347)
- Slides：`L09–L10`，尤其 `L10 p.12,22–32`，以及 `D05`。
- 核心：TRPO 用 KL trust region 防止新策略离旧策略太远；PPO 用更容易实现的 surrogate objective 和 clipping 近似这种约束。
- 第一遍重点：先读 TRPO 的动机和 surrogate objective，再读 PPO 的 clipped objective；先跳过共轭梯度和完整单调改进证明。
- 读完检查：能否解释为什么同一批 on-policy 数据不能让策略无限更新，以及 ratio clipping 在限制什么。

## 第三阶段：最大熵 RL 与逆强化学习

### 5. Control as Inference：把控制写成概率推断

- 文献：[Levine, *Reinforcement Learning and Control as Probabilistic Inference: Tutorial and Review*](https://arxiv.org/abs/1805.00909)
- Slides：`L11–L13`，尤其 `L13 p.12`。
- 核心：引入“最优性”随机变量后，最大熵 RL 可以视为图模型中的推断；随机动力学下需要变分推断，这也解释了 soft Bellman backup。
- 第一遍重点：图模型、maximum-entropy objective、soft value/Q recursion；复杂推导可跟着 slides 对照读。
- 读完检查：能否解释熵项不仅是“鼓励探索”，还改变了最优策略的概率解释。

### 6. SAC：最大熵思想的实用 actor-critic

- 文献：[Haarnoja et al., *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor*](https://proceedings.mlr.press/v80/haarnoja18b.html) → [*Soft Actor-Critic Algorithms and Applications*](https://arxiv.org/abs/1812.05905)。前者是 2018 ICML 原论文，后者是改进版本与自动温度调节的来源。
- 出处：`L13 p.16`、`D04 p.4–5`；`HW3 p.5` 链接原论文，`p.6` 明确指定第二篇第 5 节；`HW5 p.2` 指定第二篇第 6 节。
- 核心：SAC 同时最大化回报和策略熵，并结合 replay buffer 做 off-policy actor-critic，兼顾样本效率和稳定性。
- 第一遍重点：soft policy evaluation、soft policy improvement、actor 的 KL 目标；做 HW3 自动温度实验前补第二篇第 5 节。原论文的显式 V 网络版本与 D04 的现代双 Q 版本要分开看。
- 读完检查：能否把 SAC 的 critic、actor、temperature 三种更新分别说清楚。**三类 loss 不代表三个网络头**：D04 的版本有一个 actor、两个 Q critic；温度 α 通常是可学习标量，target Q 由参数平均更新。更新 critic 时 target 停止梯度；更新 actor 时通过动作对 Q 求导但不更新 critic 参数；更新温度时策略的 log-prob 停止梯度。

### 7. GAIL：直接匹配专家的访问分布

- 文献：[Ho & Ermon, *Generative Adversarial Imitation Learning*](https://proceedings.neurips.cc/paper/2016/hash/cc7e2b878868cbae992d1fb743995d8f-Abstract.html)
- Slides：`L13 p.30–34`，`L14 p.9–11`。
- 核心：判别器区分专家与策略的 state-action occupancy，策略则尝试骗过判别器；它绕过显式 reward recovery，但训练时仍需要环境交互。
- 第一遍重点：occupancy measure、式 (15)/(16)、算法 1；GAN 的一般理论不是本课程重点。
- 读完检查：能否说明 GAIL 与普通行为克隆分别在哪个分布层面做匹配。

### 8. RLHF：把语言模型放进 RL 框架

- 入门材料：[Lambert et al., *Illustrating Reinforcement Learning from Human Feedback*](https://huggingface.co/blog/rlhf)（明确图源是 `L01 p.23`、`L25 p.10`）。随后读 [Ouyang et al., *Training Language Models to Follow Instructions with Human Feedback*](https://arxiv.org/abs/2203.02155)，这是 `D07 p.3–4` 正式引用的 InstructGPT 原论文。
- Slides：`L14 p.15–38`，`L01 p.23`。
- 核心：先做 instruction/SFT，再用成对偏好训练 Bradley–Terry reward model，最后以 KL 正则约束策略模型并用 policy gradient 优化。
- 第一遍重点：prompt、completion、reward model、reference model 各自扮演什么角色；把 token 级动作和序列末端奖励画成一条轨迹。
- 读完检查：能否解释为什么要有 reference-model KL，以及 reward hacking 从哪里来。

### HW4 与语言模型项目：GRPO 及后续方法

先读 PPO 和 InstructGPT，再读以下四篇。`HW4 p.5` 明确将它们列为 optional reading；这里的阅读优先级是学习建议，不改写作业要求。

| 阅读顺序 | 论文 | 本地出处与重点 |
| --- | --- | --- |
| 1 | [Shao et al., *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*](https://arxiv.org/abs/2402.03300) | D07 p.3–4、HW4 p.5、L14 p.28 的 GRPO；看同一 prompt 多个 completion 的相对优势和 PPO clipping |
| 2 | [Liu et al., *Understanding R1-Zero-Like Training: A Critical Perspective*（DrGRPO）](https://arxiv.org/abs/2503.20783) | HW4 p.5；final LLM README 的 `drgrpo`/`dr_grpo`；看长度归一化、奖励标准差归一化带来的偏差 |
| 3 | [Yu et al., *DAPO: An Open-Source LLM Reinforcement Learning System at Scale*](https://arxiv.org/abs/2503.14476) | HW4 p.5 选读；比较 clipping、动态采样与 token 级聚合；不是本地项目必实现项 |
| 4 | [Zheng et al., *Group Sequence Policy Optimization*（GSPO）](https://arxiv.org/abs/2507.18071) | HW4 p.5、final LLM README；比较 token 与 sequence importance ratio |

完成检查：能分别指出旧策略 `π_old` 与参考策略 `π_ref` 的作用、advantage 的分组单位、loss 的 token/sequence 聚合单位。先把本地 HW4 的公式读清楚，再比较论文变体。

语言模型项目还需 DPO、IPO、AOT，见后文补充书目 F；这三篇是按 README/实现补入的背景，不是项目概述 PDF 中的正式参考文献。

## 第四阶段：模型学习与离线 RL

### 9. Dyna：真实经验与模型生成经验统一训练

- 文献：[Sutton, *Integrated Architectures for Learning, Planning, and Reacting Based on Approximating Dynamic Programming*](https://mlanthology.org/icml/1990/sutton1990icml-integrated/)
- Slides：`L16 p.21–24`。
- 核心：真实 transition 同时更新价值函数和环境模型；模型再产生额外 transition 做 planning，使 model-free learning 与 planning 使用同一种更新。
- 第一遍重点：Dyna-Q 框图和伪代码；这篇很短，建议完整读完。
- 读完检查：能否区分“从真实环境采样”和“从学习到的模型采样”的两条更新路径。

### 10. MBPO：模型 rollout 要短而多

- 文献：[Janner et al., *When to Trust Your Model: Model-Based Policy Optimization*](https://proceedings.neurips.cc/paper/2019/hash/5faf461eff3099671ad63c6f3f094f7f-Abstract.html)
- Slides：`L16 p.23–25`。
- 核心：长 rollout 会累积模型误差；MBPO 从真实 replay buffer 的状态出发，生成大量短 rollout，再交给 SAC 学习。
- 第一遍重点：branched rollout 图、算法 2、rollout length 的消融实验；误差界先看结论。
- 读完检查：能否解释为什么“一步模型不准”不代表“模型完全没用”。

### HW5 起点：先读 TD3+BC，再读 IQL/CQL

- [Fujimoto & Gu, *A Minimalist Approach to Offline Reinforcement Learning*](https://arxiv.org/abs/2106.06860)（TD3+BC）：`D09 p.2,5`、`HW5 p.1,11`。先理解“最大化 Q + 不偏离数据动作”的 actor 目标。
- HW5 实现的是 **SAC+BC**，它借用 TD3+BC 的行为约束思路；不是让你完整复现 TD3+BC。DDPG、TD3 的原论文也在 D09/HW5 的 References 中，见补充书目 B。
- 接着读 AWR，再读 IQL，会更容易理解 advantage-weighted BC 为什么能作为策略提取步骤。

### 11. IQL：不查询数据外动作

- 文献：[Kostrikov et al., *Offline Reinforcement Learning with Implicit Q-Learning*](https://openreview.net/forum?id=EblVBDNalKu)
- Slides：`L17–L18`，尤其 `L18 p.12–15`；`D09 p.3–4`、`HW5 p.4–5,11`。
- 核心：用 expectile regression 从数据内动作隐式估计较优 value，再做 Q backup，最后用 advantage-weighted behavioral cloning 提取策略；训练 critic 时不需要评价 OOD action。
- 第一遍重点：三个损失 `L_V`、`L_Q`、`L_π` 及其更新顺序。
- 读完检查：能否解释 expectile 参数增大时为什么更接近“数据支持内的最大值”。

### 12. CQL：主动压低不可信动作的 Q 值

- 文献：[Kumar et al., *Conservative Q-Learning for Offline Reinforcement Learning*](https://proceedings.neurips.cc/paper/2020/hash/0d2b2061826a5df3221116a5085a6052-Abstract.html)
- Slides：`L18 p.17–21`；`D09 p.4–5`。
- 核心：在 Bellman loss 外加入 conservative regularizer，压低策略/广泛采样动作的 Q，同时维持数据动作的 Q，从而缓解 OOD 过估计。
- 第一遍重点：CQL(H) 目标和直觉；先跳过所有理论下界细节。
- 读完检查：能否对比 IQL 的“从不查询 OOD action”和 CQL 的“查询但让估计悲观”。

### HW5 生成策略主线：Diffusion-QL → FQL → OGBench

- [Wang et al., *Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning*](https://arxiv.org/abs/2208.06193)（Diffusion-QL）：`HW5 p.6,11` 引用。看生成策略如何同时兼顾行为分布与高 Q 动作。HW5 的 FBRAC 是 flow 版本的教学铺垫，不能直接等同于这篇的 diffusion 算法。
- [Park et al., *Flow Q-Learning*](https://arxiv.org/abs/2502.02538)：`L18 p.30`、`L22 p.60`、`HW5 p.6–9,11`；优先看第 3 节和 HW5 的网络图。BC flow 学数据分布，一步 actor 做 Q 优化与蒸馏，critic 做 Bellman 回归。分别追踪三个优化对象及 stop-gradient。
- [Park et al., *OGBench: Benchmarking Offline Goal-Conditioned RL*](https://arxiv.org/abs/2410.20092)：`HW5 p.1,11`。读任务、数据收集、评价协议；本地作业调用 `singletask` 环境，不能仅凭论文题目就以为作业要求实现 goal-conditioned agent。
- 读完检查：能画出 FBRAC 与 FQL 的梯度路径，解释 FQL 为什么避免对完整 ODE 采样过程做 Q-loss 的反向传播；能解释 HW5 明确使用的双 Q 平均值与 SAC 常见最小值的差别。

### 选读：模型式离线 RL

- [Yu et al., *MOPO*](https://proceedings.neurips.cc/paper_files/paper/2020/hash/a322852ce0df73e204b7e67cbbef0d0a-Abstract.html)（`L18 p.33–34`）：在 learned model 中把不确定性作为 reward penalty。
- `MOReL` 与 `COMBO`（`L18 p.34–35`）：继续比较“构造悲观 MDP”和“对模型生成样本做 conservative Q-learning”。先读完 MBPO、IQL、CQL 再看。

## 第五阶段：探索与理论

### 13. Pseudo-count → RND：把新颖性变成奖励

- 文献：[Bellemare et al., *Unifying Count-Based Exploration and Intrinsic Motivation*](https://proceedings.neurips.cc/paper_files/paper/2016/hash/afda332245e2af431fb7b672a68b659d-Abstract.html) → [Burda et al., *Exploration by Random Network Distillation*](https://openreview.net/forum?id=H1lJJnR5Ym)
- Slides：`L19 p.18–23`。
- 核心：pseudo-count 用密度模型把大状态空间中的“见过几次”推广到相似状态；RND 则用预测一个固定随机网络的误差作为更简单的新颖性信号。
- 第一遍重点：两种 bonus 分别如何计算，以及“预测误差高”为什么不总等于“有用的新状态”。
- 读完检查：能否列出 intrinsic reward 被噪声电视或随机环境欺骗的原因。

### 14. RL Theory：用误差传播理解算法

- 资料：[Agarwal et al., *Reinforcement Learning: Theory and Algorithms*](https://rltheorybook.github.io/)
- Slides：`L20 p.12–28`，`L22 p.68–73`。
- 核心：课程这里关心的不是背定理，而是理解采样误差、函数逼近误差如何经 Bellman backup 和 horizon 放大。
- 第一遍重点：只跟着 `L20` 对应章节读 concentration、simulation lemma、fitted Q-iteration error propagation；不要从第一页顺序通读整本书。
- 读完检查：能否解释误差界里的 `1/(1-γ)` 为什么可以看作有效 horizon。

### 选读：无奖励技能与困难探索

- [Eysenbach et al., *Diversity Is All You Need*](https://openreview.net/forum?id=SJx63jRqFm)（`L23 p.10–13`）：用 skill 与访问状态之间的互信息学习可区分技能。
- [Ecoffet et al., *First Return, Then Explore*](https://www.nature.com/articles/s41586-020-03157-9)（`L23 p.18`）：Go-Explore 显式记住、返回有希望的状态，再从那里继续探索，针对 detachment 和 derailment。

## 第六阶段：多任务、迁移与层次化 RL

### 15. UVFA → HER：把失败轨迹重新解释成成功

- 文献：[Schaul et al., *Universal Value Function Approximators*](https://proceedings.mlr.press/v37/schaul15.html) → [Andrychowicz et al., *Hindsight Experience Replay*](https://proceedings.neurips.cc/paper/2017/hash/453fadbd8a1a3af50a9df4df899537b5-Abstract.html)
- Slides：`L24 p.14–19`。
- 核心：UVFA 把 goal 也作为 value function 的输入；HER 将没有达到原目标的轨迹，按实际达到的状态重标为另一个目标的成功经验。
- 第一遍重点：`V(s,g)`/`Q(s,a,g)`、goal relabeling、为什么必须使用 off-policy 方法。
- 读完检查：能否说明 HER 改的是数据标签，而不是环境真实发生的 transition。

### 16. Successor Representation → Successor Features：拆开动力学与奖励

- 文献：[Dayan, *The Successor Representation*](https://www.gatsby.ucl.ac.uk/~dayan/papers/d93b.pdf) → [Barreto et al., *Successor Features for Transfer in Reinforcement Learning*](https://proceedings.neurips.cc/paper_files/paper/2017/hash/350db081a661525235354dd3e19b8c05-Abstract.html)
- Slides：`L24 p.21–33`。
- 核心：successor representation 预测未来状态占用；successor features 将 dynamics-dependent features 与 reward weights 分离，再用 generalized policy improvement 在新奖励任务间迁移。
- 第一遍重点：分解 `Q = ψᵀw` 的含义和 GPI；证明可以第二遍再读。
- 读完检查：能否指出它适合“动力学相同、奖励变化”的任务，也能说明此前提何时不成立。

### 选读：层次化 RL

- [Bacon et al., *The Option-Critic Architecture*](https://ojs.aaai.org/index.php/AAAI/article/view/10916)（`L24 p.37`）：端到端学习 option policy 与 termination。
- [Nachum et al., *Data-Efficient Hierarchical Reinforcement Learning*](https://proceedings.neurips.cc/paper_files/paper/2018/hash/e6384711491713d29bc63fc5eeb5ba4f-Abstract.html)（`L24 p.39`）：HIRO 用高层目标和 off-policy correction 训练两层策略。
- 先读 UVFA/HER，再读 Option-Critic，最后读 HIRO；否则高层 action、subgoal 与低层策略容易混在一起。

## 第七阶段：现代应用与开放问题

这些论文用于看“前面的方法如何组合到真实前沿问题”，不适合作为入门起点：

1. [Black et al., *Training Diffusion Models with Reinforcement Learning*](https://arxiv.org/abs/2305.13301)（`L01 p.24`，`L18 p.28`）：把扩散去噪步骤视为多步决策，用 DDPO 直接优化不可微下游奖励。
2. [Gupta et al., *Reset-Free Reinforcement Learning via Multi-Task Learning*](https://arxiv.org/abs/2104.11203)（`L25 p.28`）：让不同任务互相充当 reset，使机器人能较少依赖人工复位地持续学习。
3. [Hong et al., *Zero-Shot Goal-Directed Dialogue via RL on Imagined Conversations*](https://arxiv.org/abs/2311.05584)（`L25 p.38–39`）：先让 LLM 模拟大量对话，再用离线 RL 从“合理但不一定最优”的交互中学习多轮目标导向策略。

推荐顺序是 DDPO → Reset-Free RL → Imagined Conversations：先看生成模型上的标准 policy optimization，再看真实世界 continual interaction，最后看模型生成数据与 offline RL 的组合。

## 建议的实际节奏

- 每学完 1–2 讲 slides，读对应的一篇主线论文，不要先把整份文献表读完再做作业。
- 每篇第一遍控制在 30–60 分钟，只写一页笔记：问题、关键公式、算法流程、一个优点、一个局限。
- 遇到公式卡住时先回到 slides；遇到实现卡住时再看论文伪代码和仓库代码。
- 从当前进度切入：正在做 HW3/SAC，就先读 D04 p.4–5 → SAC 两篇 → HW3 p.5–6；不需要为补书目重学整门课。

## 补充书目：把原路线略过的引用逐篇展开

以下与前面的主线合起来构成书目。顺序按知识依赖安排；“扩展”表示阅读优先级，不表示它没有出现在课程材料中。只在材料里出现方法名而没有论文题目的，标为“方法名对应”；出处不明确的放在末尾待确认表。

### A. 前置、行为克隆与生成模型

| 文献 | 出处 | 阅读位置与重点 |
| --- | --- | --- |
| Kingma & Ba, [*Adam: A Method for Stochastic Optimization*](https://arxiv.org/abs/1412.6980) | HW1 p.2,4；作业书目只写 Kingma | 工具背景；了解一、二阶矩估计和 bias correction，不必作为 RL 论文精读 |
| Pomerleau, [*ALVINN: An Autonomous Land Vehicle in a Neural Network*](https://publications.ri.cmu.edu/alvinn-an-autonomous-land-vehicle-in-a-neural-network) | L02 p.10 的历史案例（课件标题略写） | BC 前后浏览；早期端到端驾驶，不是现代 DAgger |
| Bojarski et al., [*End to End Learning for Self-Driving Cars*](https://arxiv.org/abs/1604.07316) | L02 p.15–16：作者、年份与 NVIDIA 视频，对应论文 | 案例；比较示范数据、增强和闭环控制 |
| Shah et al., [*GNM: A General Navigation Model to Drive Any Robot*](https://arxiv.org/abs/2210.03370) | L03 p.31 | BC 之后；跨平台、目标条件导航 |
| Higgins et al., [*β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework*](https://openreview.net/forum?id=Sy2fzU9gl) | L12 p.13：作者与年份，对应 β-VAE 工作 | 扩展；读过 ELBO 后比较 KL 权重与表示学习 |
| Kingma & Welling, [*Auto-Encoding Variational Bayes*](https://arxiv.org/abs/1312.6114) | **补充背景**：L11–12、D06 的 VAE/重参数化，没有正式文献引用 | 只补 ELBO 与重参数化；不是作业新增要求 |
| Lipman et al., [*Flow Matching for Generative Modeling*](https://arxiv.org/abs/2210.02747) | **补充背景**：L03、HW1、HW5 的 flow matching | 先读 Roelants 教程，再看 conditional flow matching 目标；HW1 正式 References 只有 Diffusion Policy 和 Adam |

### B. 策略梯度、连续控制与 Q 学习

本阶段优先级为 GAE → DQN/Double DQN → TRPO/PPO → SAC 两篇；以下补齐它们的背景与 D09/HW5 的引用。

| 文献 | 出处 | 阅读位置与重点 |
| --- | --- | --- |
| Williams, *Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning*（REINFORCE） | **补充背景**：L05、HW2、HW4 使用该方法，未列这篇正式书目 | 看 score-function estimator 与 baseline 无偏性；可用 L05 推导代替原文首读 |
| Mnih et al., [*Playing Atari with Deep Reinforcement Learning*](https://arxiv.org/abs/1312.5602) | L04 p.40、L08 p.7、L22 p.7 的 Mnih '13 | DQN 2013 版；与主线 2015 Nature 版区分，先读后者 |
| Lillicrap et al., [*Continuous Control with Deep Reinforcement Learning*](https://arxiv.org/abs/1509.02971)（DDPG） | D09 p.1,5；HW5 p.1,11 | 扩展；连续动作 actor 替代离散 argmax，理解确定性 actor 的更新 |
| Fujimoto et al., [*Addressing Function Approximation Error in Actor-Critic Methods*](https://proceedings.mlr.press/v80/fujimoto18a.html)（TD3） | D04 p.4 点名；D09 p.1,5；HW5 p.1,11 | DDPG 后；双 critic、延迟 actor 更新、target policy smoothing；再接 TD3+BC |

### C. Control as inference、逆 RL 与对抗模仿

主线 Levine 教程 → 最大熵 IRL → Guided Cost Learning → GAN/IRL 联系 → GAIL；只为理解 GAIL 时不必把所有 GAN 图注都精读。

| 文献 | 出处 | 阅读位置与重点 |
| --- | --- | --- |
| Ziebart et al., [*Maximum Entropy Inverse Reinforcement Learning*](https://www.cs.cmu.edu/~bziebart/publications/maximum-entropy-inverse-reinforcement-learning.html) | L12 p.21 的 Ziebart '08 | 扩展；轨迹分布、特征匹配与 partition function；与最大熵正向 RL 区分 |
| Finn et al., [*Guided Cost Learning: Deep Inverse Optimal Control via Policy Optimization*](https://proceedings.mlr.press/v48/finn16.html) | L13 p.27 | 最大熵 IRL 后；用采样近似配分函数，交替更新 reward 与 policy |
| Finn, Christiano et al., [*A Connection between Generative Adversarial Networks, Inverse Reinforcement Learning, and Energy-Based Models*](https://arxiv.org/abs/1611.03852) | L13 p.31–32；L22 p.32 | Guided Cost Learning 后、GAIL 前；reward/energy 与 discriminator 的关系 |
| Goodfellow et al., [*Generative Adversarial Nets*](https://arxiv.org/abs/1406.2661) | L13 p.30、L14 p.10 | GAIL 的生成器/判别器背景 |
| Zhu et al., [*Unpaired Image-to-Image Translation Using Cycle-Consistent Adversarial Networks*](https://arxiv.org/abs/1703.10593)（CycleGAN） | L13 p.30、L14 p.10 图注 | 图像生成案例，GAIL 首读可跳过 |
| Arjovsky et al., [*Wasserstein GAN*](https://arxiv.org/abs/1701.07875) | L13 p.30、L14 p.10 图注 | 扩展；分布距离与对抗训练 |
| Isola et al., [*Image-to-Image Translation with Conditional Adversarial Networks*](https://arxiv.org/abs/1611.07004)（pix2pix） | L13 p.30、L14 p.10 图注 | 条件 GAN 案例 |
| Hausman et al., [*Multi-Modal Imitation Learning from Unstructured Demonstrations Using Generative Adversarial Nets*](https://arxiv.org/abs/1705.10479) | L13 p.34、L14 p.12，按完整作者组对应 | GAIL 后；无结构、多模态示范与技能区分 |
| Peng et al., [*SFV: Reinforcement Learning of Physical Skills from Videos*](https://arxiv.org/abs/1810.03599) | L13 p.34、L14 p.12 的 Peng/Kanazawa/Toyer/Abbeel/Levine 作者组对应 | 视频模仿案例；不要误配为作者组不同的 DeepMimic |

### D. 学模型、规划与模型式离线 RL

Dyna 和 MBPO 已在主线展开。先分清 epistemic/aleatoric uncertainty，再读模型 rollout 如何接入价值学习；MOPO/MOReL/COMBO 放在 IQL/CQL 之后。

| 文献 | 出处 | 阅读位置与重点 |
| --- | --- | --- |
| Blundell et al., [*Weight Uncertainty in Neural Networks*](https://arxiv.org/abs/1505.05424)（Bayes by Backprop） | L15 p.19；D08 p.2,4 | VI 后；对权重建模来表达 epistemic uncertainty |
| Gal et al., [*Concrete Dropout*](https://arxiv.org/abs/1705.07832) | L15 p.19 | 不确定性扩展；可学习 dropout 概率 |
| Karaletsos & Bui, [*Hierarchical Gaussian Process Priors for Bayesian Neural Network Weights*](https://proceedings.nips.cc/paper/2020/hash/c70341de2c112a6b3496aec1f631dddd-Abstract.html) | D08 p.2,4，原路线遗漏 | BNN 扩展；不必在实现 ensemble 前读完 |
| Nagabandi et al., [*Deep Dynamics Models for Learning Dexterous Manipulation*](https://arxiv.org/abs/1909.11652) | L16 p.13 | 规划案例；学习动力学、模型集合与动作序列优化 |
| Parmas et al., [*PIPPS: Flexible Model-Based Policy Search Robust to the Curse of Chaos*](https://proceedings.mlr.press/v80/parmas18a.html) | L16 p.17 写作 “PIPP” | 扩展；对比 pathwise gradient 与 likelihood-ratio gradient 的稳定性 |
| Gu et al., [*Continuous Deep Q-Learning with Model-Based Acceleration*](https://arxiv.org/abs/1603.00748)（NAF/MBA） | L16 p.25、L22 p.42；D08 p.4 | Dyna 后；模型数据如何加速连续动作价值学习 |
| Feinberg et al., [*Model-Based Value Estimation for Efficient Model-Free Reinforcement Learning*](https://arxiv.org/abs/1803.00101)（MVE） | L16 p.25、L22 p.42 写 Model-Based Value Expansion；D08 p.4 给完整题目 | MBA 后、MBPO 前；短模型展开用于 value target |
| Kidambi et al., [*MOReL: Model-Based Offline Reinforcement Learning*](https://arxiv.org/abs/2005.05951) | L18 p.34、L22 p.61 | 对比 MOPO；用模型置信程度构造悲观环境 |
| Yu et al., [*COMBO: Conservative Offline Model-Based Policy Optimization*](https://arxiv.org/abs/2102.08363) | L18 p.35 | CQL + model-based RL；对模型产生的状态动作施加保守约束 |

### E. 离线 RL、生成策略与离线到在线项目

阅读顺序：TD3+BC/SAC+BC → AWR/BRAC → IQL/CQL → XQL/Dual RL（理论扩展）；生成策略方向接 Diffusion-QL → IDQL/FQL → DSRL。AWAC、RLPD 用作离线数据如何帮助在线学习的对照。

| 文献 | 出处 | 阅读位置与重点 |
| --- | --- | --- |
| Wu et al., [*Behavior Regularized Offline Reinforcement Learning*](https://arxiv.org/abs/1911.11361)（BRAC） | L18 p.7、L22 p.51 | 先比较 actor 约束和 reward/value 约束，以及 KL 方向 |
| Peng et al., [*Advantage-Weighted Regression: Simple and Scalable Off-Policy Reinforcement Learning*](https://arxiv.org/abs/1910.00177)（AWR） | L18 p.9、L22 p.53；D09 p.2–3,5 | IQL 前；从 KL 约束推导加权最大似然 |
| Peters et al., [*Relative Entropy Policy Search*](https://ojs.aaai.org/index.php/AAAI/article/view/7727)（REPS） | L18 p.9、L22 p.53 | AWR 理论支线；对偶问题与相对熵约束 |
| Rawlik et al., [*On Stochastic Optimal Control and Reinforcement Learning by Approximate Inference*](https://www.ijcai.org/Proceedings/13/Papers/455.pdf)（ψ-learning） | L18 p.9、L22 p.53 仅写 Rawlik/psi-learning，按方法名对应 | control as inference 与 KL 正则策略改进的联系；所链为 2013 extended abstract，原工作发表于 RSS 2012 |
| Nair et al., [*AWAC: Accelerating Online Reinforcement Learning with Offline Datasets*](https://arxiv.org/abs/2006.09359) | L18 p.9–10、L22 p.53–54 | AWR 后；离线初始化和在线数据如何共同更新 actor-critic |
| Garg et al., [*Extreme Q-Learning: MaxEnt RL without Entropy*](https://arxiv.org/abs/2301.02328)（XQL） | D09 p.5，原路线遗漏 | IQL/CQL 后；隐式估计 soft value 的另一种回归 |
| Sikchi et al., [*Dual RL: Unification and New Methods for Reinforcement and Imitation Learning*](https://arxiv.org/abs/2302.08560) | D09 p.5，原路线遗漏 | XQL 后；用访问分布与对偶形式对照算法，注意等价结论的假设 |
| Ball et al., [*Efficient Online Reinforcement Learning with Offline Data*](https://arxiv.org/abs/2302.02948)（RLPD） | L18 p.26 | 离线到在线基线；混合 offline/online minibatch，不等同于先离线预训练 |
| Hansen-Estruch et al., [*IDQL: Implicit Q-Learning as an Actor-Critic Method with Diffusion Policies*](https://arxiv.org/abs/2304.10573) | L18 p.29、L22 p.59 | IQL 与生成策略后；分开行为模型、value learning 与动作重采样 |
| Ren et al., [*Diffusion Policy Policy Optimization*](https://arxiv.org/abs/2409.00588)（DPPO） | L18 p.28 | 与 DDPO 对照；用 policy gradient 微调 diffusion policy |
| Wagenmaker et al., [*Steering Your Diffusion Policy with Latent Space Reinforcement Learning*](https://arxiv.org/abs/2506.15799)（DSRL） | L18 p.31；项目 `dsrl_agent.py` 类注释有原文链接 | 先读 SAC 和 flow/diffusion；把潜在噪声作为 RL 动作来引导生成策略 |
| Psenka et al., [*Learning a Diffusion Model Policy from Rewards via Q-Score Matching*](https://arxiv.org/abs/2312.11752)（QSM） | **补充背景**：项目 `qsm_agent.py` 有 QSMAgent/DDPM 框架，无正式论文链接 | 对照 score 与 Q 对动作的梯度；代码有 TODO，不能声称已经实现完整原论文 |
| IFQL：本地 IQL + flow 行为策略 + best-of-N 动作选择 | 项目 `ifql_agent.py` 的注释、接口与 TODO | **代码路线，不虚构同名独立论文**；先读 IQL → IDQL → HW1 flow。与 FQL 的一步蒸馏 actor 分清 |

本地离线到在线项目目录只有 `problem/README.md` 与代码，没有发现专门的项目说明 PDF。SAC+BC、FQL 复用 HW5 背景；DSRL 有明确原文链接；QSM、IFQL 的阅读对应关系按上述证据等级处理。

### F. LLM 的监督微调与偏好学习项目

InstructGPT、DeepSeekMath、DrGRPO、DAPO、GSPO 已在主线列出。偏好学习支线先看 Bradley–Terry/KL 正则，再读 DPO → IPO → AOT；不必先完成机器人 RL 的所有扩展。

| 文献 | 出处 | 阅读位置与重点 |
| --- | --- | --- |
| Zhang, Chosen & Andreas, [*Unforgettable Generalization in Language Models*](https://arxiv.org/abs/2409.02228) | L14 p.16；课件标 Andreas et al. 2022，按题目匹配的原文为 2024 | SFT 背景；观察错误标签与行为格式学习的区别 |
| Chung et al., [*Scaling Instruction-Finetuned Language Models*](https://arxiv.org/abs/2210.11416) | L14 p.17 | InstructGPT 前；指令微调的数据与泛化 |
| Rafailov et al., [*Direct Preference Optimization: Your Language Model Is Secretly a Reward Model*](https://arxiv.org/abs/2305.18290)（DPO） | **补充背景**：final LLM README 的 `dpo`，`offline/losses.py` 的 DPO 分支 | 从 KL 正则最优策略关系消去显式 reward model；区分 reference 与 trainable policy |
| Azar et al., [*A General Theoretical Paradigm to Understand Learning from Human Preferences*](https://arxiv.org/abs/2310.12036)（IPO） | **补充背景**：同 README 的 `ipo`、同代码的 IPO 分支 | DPO 后；看 pairwise preference 目标与有限目标间隔 |
| Melnyk et al., [*Distributional Preference Alignment of LLMs via Optimal Transport*](https://arxiv.org/abs/2406.05882)（AOT） | **补充背景**：同 README 的 `aot`、同代码的 AOT 分支 | DPO/IPO 后；批内排序、分布偏好与最优传输，不要直接等同逐对 DPO |

项目的 `final_project_outline.pdf` 是通用项目安排、proposal/report 要求，不包含上述方法的正式参考文献表；具体方法列表来自 [项目 README](homework_spring2026/final_project_llm_rl/README.md) 及 `offline/losses.py`、`rl/`。README 中的方法名不代表模板里的 TODO 已完成。

### G. 探索、目标条件 RL、层次与元学习

| 文献 | 出处 | 阅读位置与重点 |
| --- | --- | --- |
| Strehl & Littman, [*An Analysis of Model-Based Interval Estimation for Markov Decision Processes*](https://www.sciencedirect.com/science/article/pii/S0022000008000767)（MBIE-EB） | L19 p.19 的作者、年份与方法 | count bonus 理论扩展；为什么访问次数能变成乐观奖励 |
| Kolter & Ng, *Near-Bayesian Exploration in Polynomial Time*（BEB） | L19 p.19 | 对比 `1/N` 与 `1/√N` 类型 bonus；先看设定再比较结论 |
| Fu et al., [*EX2: Exploration with Exemplar Models for Deep Reinforcement Learning*](https://arxiv.org/abs/1703.01260) | L19 p.21 方法名 | pseudo-count 后；用判别式 exemplar 模型衡量新颖性 |
| Gregor et al., [*Variational Intrinsic Control*](https://arxiv.org/abs/1611.07507)（VIC） | L23 p.13 | DIAYN 前后；技能和可控结果间的互信息 |
| Nair et al., [*Visual Reinforcement Learning with Imagined Goals*](https://arxiv.org/abs/1807.04742)（RIG） | L23 p.17 | 先补 UVFA/HER 与 VAE；潜在目标与重标注 |
| Pong et al., [*Skew-Fit: State-Covering Self-Supervised Reinforcement Learning*](https://arxiv.org/abs/1903.03698) | L23 p.17（课件作者排列与正式论文不同） | RIG 后；通过目标采样扩大状态覆盖 |
| Misra et al., [*Kinematic State Abstraction and Provably Efficient Rich-Observation Reinforcement Learning*](https://proceedings.mlr.press/v119/misra20a.html)（HOMER） | L23 p.18 | 理论扩展；隐状态抽象与探索覆盖，先具备 L20 背景 |
| Schaul et al., [*Ray Interference: A Source of Plateaus in Deep Reinforcement Learning*](https://arxiv.org/abs/1904.11455) | L24 p.12 | 多任务难点；任务间干扰与学习停滞 |
| Kaelbling, [*Learning to Achieve Goals*](https://mlanthology.org/ijcai/1993/kaelbling1993ijcai-learning/) | L24 p.16,19 | UVFA/HER 前后；目标条件学习的早期工作 |
| Nasiriany et al., [*Planning with Goal-Conditioned Policies*](https://arxiv.org/abs/1911.08453) | L24 p.19 | UVFA/HER 后；将 goal-conditioned policy/value 用作规划工具 |
| Eysenbach et al., [*C-Learning: Learning to Achieve Goals via Recursive Classification*](https://arxiv.org/abs/2011.08909) | L24 p.16,33 | successor representation 后；用递归分类估计未来可达性 |
| Zhou et al., [*ArCHer: Training Language Model Agents via Hierarchical Multi-Turn RL*](https://arxiv.org/abs/2402.19446) | L24 p.40 | HIRO 与 LLM RL 后；utterance/token 两个决策层次 |
| Ravi & Larochelle, [*Optimization as a Model for Few-Shot Learning*](https://openreview.net/forum?id=rJY0-Kcll) | L24 p.42 图注 | 元学习背景；先区分跨任务训练与单任务适应 |

L24 p.41–53 还讲 recurrent meta-RL/POMDP，但未明确给出 RL² 或 MAML 论文引用；不把所有相关经典论文自动算作“课件已引用”。本路线用该讲本身补齐元学习内容。

### H. 应用案例与非论文阅读

原版只用一句话带过的应用也保留出处，但优先级低于当前作业。

| 文献/资料 | 出处 | 处理方式 |
| --- | --- | --- |
| Sims, [*Evolving Virtual Creatures*](https://www.karlsims.com/papers/siggraph94.pdf) | L01 p.7 写 “Evolved Virtual Creatures” | 演化/控制历史案例，不是 deep RL 主线 |
| Silver et al., [*Mastering the Game of Go with Deep Neural Networks and Tree Search*](https://www.nature.com/articles/nature16961)（AlphaGo） | L06 p.14；L01/L17/L25 也用 AlphaGo 案例 | 价值、策略与搜索学完后再读 |
| Rudin et al., [*Advanced Skills by Learning Locomotion and Local Navigation End-to-End*](https://arxiv.org/abs/2209.12827) | L01 p.18、L25 p.8 | PPO/连续控制后的机器人案例 |
| Allshire et al., [*Visual Imitation Enables Contextual Humanoid Control*](https://arxiv.org/abs/2505.03729)（VideoMimic） | L01 p.19 | BC、模仿与 sim-to-real 案例；VideoMimic 是方法名，不是正式论文题目 |
| Google Research, [*Chip Design with Deep Reinforcement Learning*](https://ai.googleblog.com/2020/04/chip-design-with-deep-reinforcement.html) | L01 p.25，原始图源 URL | 官方博客案例；不自动替换成后来发表的芯片设计论文 |
| Peter Roelants, [Flow Matching 教程](https://peterroelants.github.io/posts/flow_matching_intro/) | L03 p.14、L21 p.12 | HW1 优先阅读的插图/教程来源，已提升到主线 |
| Lambert et al., [*Illustrating Reinforcement Learning from Human Feedback*](https://huggingface.co/blog/rlhf) | L01 p.23、L25 p.10 | RLHF 入门博客，不能替代 D07 引用的 InstructGPT 原论文 |
| [ComfyAI 的 RLHF 文章](https://comfyai.app/article/llm-posttraining/reinforcement-learning-from-human-feedback) | L14 p.38 明列图片来源 | 仅保留图源，不作为核实算法的权威论文 |
| [Fall 2023 Lecture 5](https://rail.eecs.berkeley.edu/deeprlcourse-fa23/deeprlcourse-fa23/static/slides/lec-5.pdf) | D03 p.4 的内嵌链接，指向 p.19 | baseline 的方差最小化推导；旧课程讲义，不是额外研究论文 |
| David Silver 的 exploration 讲义 | L19 p.8 原文 URL 为 `http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/XX.pdf` | 例子来源；`XX.pdf` 看起来是占位路径，不能当成已核实可用链接 |
| [Physical Intelligence](https://www.pi.website/) | L01 p.20、L25 p.9,29 写 `pi.website` | 公司/项目演示，没有足够信息指定 π0、π0.5 等某一篇论文 |
| 截断正态分布讲义 | HW3/HW5/离线到在线项目的 `src/infrastructure/distributions.py` 引用 `https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf` | 实现参考，不是 RL 主线论文；按需查概率密度/变换公式 |

### I. 未唯一定位的出处：保留线索，不冒充已经核实

| 材料线索 | 位置 | 尚不能下的结论 |
| --- | --- | --- |
| “Model-Predictive Control with iLQG”, Yuval Tassa, 2012 | L01 p.7 | 可能对应相关控制论文或演示；课件标签不足以直接确定正式题目 |
| TD-Gammon, Gerald Tesauro 1992 | L06 p.14 | Tesauro 有多篇不同年份的 TD-Gammon 文章；不能直接用 1995 论文替换 1992 图注 |
| Peters & Schaal 2008 | L10 p.29 | 同年有多个相关发表，仅作者/年份不能唯一确定图来自哪篇 |
| Mombaur et al. '09；Li & Todorov '06 | L12 p.21；Li & Todorov 也在 p.36 | 保留逆最优控制/控制推断支线，未把搜索到的相近题目当作确认出处 |
| Marjanović et al. '25 | L14 p.19 | 只有推理模型图的作者/年份，正式题目待核实 |
| BrainPort；Martinez et al.；Roe et al. | L01 p.35 | 感觉替代/可塑性图注，缺题目年份；不凭作者名填论文 |
| UW IPD；Cathy Wu；Wolpert；Moravec/Pinker；Turing；xkcd | L01 p.5,22,30–31,38；L25 p.23,34 | 图片、引言、人物或项目线索，不是已经确定的逐篇学术引用 |

## 来源覆盖表：从材料反查阅读入口

这是对本地快照的覆盖记录，不是对课程官网以后更新的保证。检查了 PDF 正文文本、References 和内嵌 URI；同时检查作业/项目 README 与代码中的文献线索。没有逐张 OCR 纯图片里的微小引文，无法唯一定位的图注见上表。**因此可说已系统补齐可识别引用，不能说所有图片出处都已 100% 核实。**

### Notes：25 / 25 份

| 来源 | 已归入的内容 |
| --- | --- |
| [L01 Introduction](<notes/01 - Introduction.pdf>) | Bitter Lesson、DQN 2015、RLHF 博客、DDPO、Rudin、VideoMimic、芯片博客、Sims；公司/历史/图注线索见 H/I |
| [L02 Behavioral Cloning](<notes/02 - Behavioral Cloning.pdf>) | DAgger、ALVINN、Bojarski 驾驶论文 |
| [L03 Behavioral Cloning Part 2](<notes/03 - Behavioral Cloning Part 2.pdf>) | Causal Confusion、Flow Matching 教程、Diffusion Policy、GNM |
| [L04 RL Basics](<notes/04 - RL Basics.pdf>) | MDP/Bellman 基础、DQN 2013；教材作前置 |
| [L05 Policy Gradients](<notes/05 - Policy Gradients.pdf>) | 策略梯度推导；REINFORCE 原论文标作补充背景 |
| [L06 Actor Critic](<notes/06 - Actor Critic.pdf>) | GAE、AlphaGo、TD-Gammon 待确认图注 |
| [L07 Value-Based RL](<notes/07 - Value-Based RL.pdf>) | 值迭代、Q-learning 主线；未发现额外明确论文引用 |
| [L08 Q-learning in Practice](<notes/08 - Q-learning in Practice.pdf>) | DQN 2013、实践技巧；Schulman slide credit 不单列为论文 |
| [L09 Advanced Policy Gradients Part 1](<notes/09 - Advanced Policy Gradients Part 1.pdf>) | importance sampling/PPO 主线 |
| [L10 Advanced Policy Gradients Part 2](<notes/10 - Advanced Policy Gradients Part 2.pdf>) | TRPO、PPO、自然梯度；Peters & Schaal 图注待确认 |
| [L11 Variational Inference](<notes/11 - Variational Inference.pdf>) | VI/ELBO；VAE 原论文标作补充背景 |
| [L12 VI in RL](<notes/12 - VI in RL.pdf>) | β-VAE、Ziebart；Mombaur/Li & Todorov 待确认 |
| [L13 Control as Inference](<notes/13 - Control as Inference.pdf>) | Levine 教程、SAC、Guided Cost Learning、GAN 四篇、GAN/IRL 联系、GAIL、Hausman、SFV |
| [L14 LLM RL](<notes/14 - LLM RL.pdf>) | GAIL/GAN 案例复用、Unforgettable Generalization、Scaling Instruction-Finetuned LMs、GRPO、ComfyAI 图源、Marjanović 待确认；POMDP 部分仍需读讲义 |
| [L15 Model-Based RL Part 1](<notes/15 - Model-Based RL Part 1.pdf>) | Bayes by Backprop、Concrete Dropout；Veo/Sora 只是模型案例标签 |
| [L16 Model-Based RL Part 2](<notes/16 - Model-Based RL Part 2.pdf>) | Deep Dynamics、PIPPS、Dyna、MBA、MVE、MBPO |
| [L17 Offline RL Part 1](<notes/17 - Offline RL Part 1.pdf>) | 离线分布偏移、策略约束；没有因此新增一套论文 |
| [L18 Offline RL Part 2](<notes/18 - Offline RL Part 2.pdf>) | BRAC、AWR、REPS、ψ-learning、AWAC、IQL、CQL、RLPD、DDPO、DPPO、IDQL、FQL、DSRL、MOPO、MOReL、COMBO；AC+BC 方法接 HW5 书目 |
| [L19 Exploration](<notes/19 - Exploration.pdf>) | pseudo-count、MBIE-EB、BEB、EX2、RND、Silver 讲义线索；CTS/随机网络/压缩长度作为方法背景，不凭术语扩列论文 |
| [L20 RL Theory](<notes/20 - RL Theory.pdf>) | Agarwal/Jiang/Kakade/Sun 理论教材、Kumar slides credit |
| [L21 Midterm Review Part 1](<notes/21 - Midterm Review Part 1.pdf>) | DAgger、Flow Matching 教程、TRPO、GAE，重复引用合并 |
| [L22 Midterm Review Part 2](<notes/22 - Midterm Review Part 2.pdf>) | DQN、GAN/IRL 联系、MBA/MVE/MBPO、BRAC/AWR/AWAC、REPS/ψ-learning、IQL/CQL、IDQL/FQL、MOPO/MOReL、pseudo-count/RND、理论教材，重复引用合并 |
| [L23 Advanced Exploration](<notes/23 - Advanced Exploration.pdf>) | DIAYN、VIC、RIG、Skew-Fit、HOMER、Go-Explore |
| [L24 Multi-task RL](<notes/24 - Multi-task RL.pdf>) | Ray Interference、Kaelbling、UVFA/HER、Planning with Goal-Conditioned Policies、SR/SF、C-Learning、Option-Critic、HIRO、ArCHer、Ravi & Larochelle；补上 meta-RL 内容入口 |
| [L25 Challenges and Open Problems](<notes/25 - Challenges and Open Problems.pdf>) | Rudin、RLHF 博客、Reset-Free RL、Imagined Conversations；其余案例/引言见 H/I |

### Discussion：10 / 10 份

| 来源 | 正式引用/相关入口 |
| --- | --- |
| [D01 PyTorch Tutorial](<discussion_sections/01 - PyTorch Tutorial.pdf>) | 工具入门；未发现额外论文参考文献 |
| [D02 Part 1 Probability Review](<discussion_sections/02 Part 1 - Probability Review.pdf>) | 概率、期望等前置；未发现额外论文引用 |
| [D02 Part 2 BC Distributional Shift](<discussion_sections/02 Part 2 - BC Distributional Shift.pdf>) | BC 分布偏移证明，与 DAgger 主线配套；未另列正式书目 |
| [D03 Policy Gradients and Actor Critic](<discussion_sections/03 - Policy Gradients and Actor Critic.pdf>) | p.4 旧 Lecture 5 链接；p.8 GAE 原论文 |
| [D04 DQN and SAC](<discussion_sections/04 - DQN and SAC.pdf>) | DQN 2015、Double DQN 2016；p.4–5 SAC 与 TD3 的 clipped double-Q 关系 |
| [D05 Advanced Policy Gradients](<discussion_sections/05 - Advanced Policy Gradients.pdf>) | TRPO/PPO、策略更新推导；没有单独 References 表 |
| [D06 Variational Inference](<discussion_sections/06 - Variational Inference.pdf>) | VI、ELBO、VAE、control as inference；没有单独 References 表 |
| [D07 IRL and LLM RL](<discussion_sections/07 - IRL and LLM RL.pdf>) | p.4 References **2 篇**：InstructGPT、DeepSeekMath |
| [D08 Model-Based RL](<discussion_sections/08 - Model-Based RL.pdf>) | p.4 References **6 篇**：Weight Uncertainty、Hierarchical GP Priors、Dyna、MBA、MVE、MBPO |
| [D09 Offline RL](<discussion_sections/09 - Offline RL.pdf>) | p.5 References **9 篇**：TD3+BC、TD3、XQL、SAC 原论文、IQL、CQL、DDPG、AWR、Dual RL |

### Homework 与两个项目

| 来源 | 引用覆盖与阅读顺序 |
| --- | --- |
| [HW1](homework_spring2026/hw1/hw1.pdf) | p.4 References **2 篇**：Diffusion Policy、Adam。先 BC → flow 教程 → Diffusion Policy 的表达动机；Flow Matching 原论文是补充背景 |
| [HW2](homework_spring2026/hw2/hw2.pdf) | p.3 GAE 链接；先 L05/D03 → baseline → GAE。没有末尾独立 References 表 |
| [HW3](homework_spring2026/hw3/hw3.pdf) | p.5 SAC 原论文、p.6 SAC Algorithms and Applications；DQN/Double DQN 配 D04。不要漏自动温度论文 |
| [HW4](homework_spring2026/hw4/hw4.pdf) | p.5 明列 **4 篇选读**：DeepSeekMath、DrGRPO、DAPO、GSPO；先 PG/GAE/TRPO/PPO，再接 LLM RL |
| [HW5](homework_spring2026/hw5/hw5.pdf) | p.11 References **9 篇**：TD3+BC、TD3、SAC 两篇、IQL、DDPG、OGBench、FQL、Diffusion-QL。三部分实现路线为 SAC+BC → IQL → FQL |
| [LLM 项目概述](homework_spring2026/final_project_llm_rl/final_project_outline.pdf) 与 [README](homework_spring2026/final_project_llm_rl/README.md) | PDF 是通用项目安排；README/代码：DPO、IPO、AOT、reward model、REINFORCE/GRPO/DrGRPO/GSPO，以及 online extension 接口 |
| [Offline-to-online 项目 README](homework_spring2026/final_project_offline_online/problem/README.md) 与 `src/agents/` | SAC+BC、FQL、IFQL、QSM、DSRL；论文来源/推断程度见 E，未发现该项目独立 PDF |

排查时也检查了 HW1–5 README、项目 README 和源文件中的 URL/文献提及。环境安装、W&B、Modal、PyTorch、仓库入口、会议格式要求等链接属于工具/行政资料；不把它们扩写成论文。个人实验报告、训练曲线、数据集对话样本、锁文件中的依赖元数据不作为课程指定阅读来源。

## 本次排查修正了什么

- 从“notes 的精选主线”扩为 notes + discussion + HW + final projects 的可追溯路线。
- 补齐 GAE、Double DQN、SAC 第二篇、HW4 四篇选读、TD3+BC/DDPG/TD3、Diffusion-QL/FQL/OGBench、D08 的 GP-prior 文献、D09 的 XQL/Dual RL。
- 将原先仅一串缩写带过的 BRAC/AWR/AWAC/RLPD/IDQL、模型不确定性、探索、goal-conditioned RL 等逐篇展开。
- 给 Flow Matching、生成策略与语言模型项目补上前置和阅读分支；把正式引用与依据代码补充的背景区分开。
- 修正 SAC 对应页为 L13 p.16；区分两篇 SAC、Diffusion Policy 的多个发表年份、PIPPS/MVE/VideoMimic 的课件简称与正式题目。
- 保留未唯一定位图注的清单；外部链接只对新增/歧义重点条目查了原论文或作者页面，另对 arXiv 条目批量核对了页面题目；部分请求超时或返回代理 502，**不把这些条目标作链接失效，也不声称全部外链均已通过可访问性检查**。
