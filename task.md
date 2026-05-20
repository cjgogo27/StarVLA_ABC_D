创智夏季营 · 具⾝智能⽅向实训考题
赛题：VLA/VA 模型⻓程操控基准测评
⼀、背景
具⾝智能（Embodied AI）是当前 AI 领域最具挑战性的前沿⽅向之⼀。Vision-Language-Action
(VLA) 模型和 World Model (WM) 是两种主流技术路线，分别以“感知-语⾔-动作”端到端映射和
“世界模型驱动动作预测”为核⼼理念。
StarVLA 是⼀个开源、模块化的具⾝智能框架（GitHub），⽀持 VLA 和 WM 两种路线下的多种模型架
构（FAST、OFT、PI、GR00T 等），并提供统⼀的数据接⼝、训练流程和评测⼯具链。
CALVIN ABC→D 是⻓程机器⼈操控领域的标准基准，要求模型在训练环境 A/B/C 上学习，在未⻅过的
环境 D 上完成 5 个连续任务链。该基准全⾯考察模型的泛化能⼒、⻓程规划能⼒和误差累积鲁棒性。
⼆、任务描述
参赛选⼿需基于 StarVLA 框架，选择合适的模型架构和训练策略，在 CALVIN ABC→D 基准上实现并
测评⼀个 Action Policy Model。考核时间为 2 天。
三、具体要求
3.1 模型选择与实现
基座模型：选择以下之⼀作为视觉-语⾔⻣⼲：
• Qwen3.5 系列（0.8B / 2B / 4B / 9B）——最新多模态⼤模型，StarVLA 已原⽣⽀持
• CosmoPredict2（2B）⸺NVIDIA 世界模型路线，基于视频⽣成 DiT 架构
动作头架构：在 StarVLA ⽀持的以下架构中选择或组合：
• FAST：⾃回归离散 Action Token
• OFT：并⾏连续动作 MLP 头
• PI：Flow-Matching 扩散动作头
• GR00T：双系统架构（VLM System 2 + Flow-Matching System 1）
⿎励创新：在 StarVLA 框架下实现新架构（如 MOT、Mixture-of-Experts 动作头、⾃适应规划模块
等），有额外加分。
3.2 数据与训练
预训练：⿎励 于公开数据集进⾏预训练，可选数据集包括但不限于：
• Open X-Embodiment (OXE)
• Bridge (WidowX)
• Fractal / RT-1
• LIBERO
• RoboCasa / RoboTwin 2.0
数据增强：⿎励采⽤数据增强策略（视觉增强、轨迹扰动、语⾔指令改写等）或⾃⾏构建更多训练数
据。
后训练：基于 CALVIN 环境中的特定 Failure Pattern 进⾏针对性后训练（如 RL Post-Training、
Failure-Aware Finetuning 等）。
3.3 测评与分析
在 CALVIN ABC→D split 上进⾏完整测评，报告以下指标：
• Task 1 ~ Task 5 各阶段成功率
• 平均任务链⻓度（Average Chain Length，满分 5.0）
技术路线论证：给出选择 VLA 或 WM 路线的原因，从以下维度分析：
• 数据效率
• 泛化能⼒（ABC→D 跨环境）
• ⻓程规划能⼒
• 计算成本与推理效率
Failure Pattern 分析：
• 记录并分类典型失败案例（如误差累积、环境泛化失败、⻓程规划退化、动作精度不⾜等）
• 分析失败根因
• 提出并实施改进⽅案（可以是后训练、架构改进、推理策略等）
四、交付物
1. 代码仓库：Fork StarVLA 并提交所有修改代码，含清晰的 README
2. 模型权重：上传⾄ HuggingFace 或提供下载链接
3. 测评报告：包含完整的 CALVIN ABC→D 测评结果（Task 1~5 成功率 + 平均链⻓）
4. 技术⽂档：技术路线选择论证 + Failure Pattern 分析报告
5. 答辩 PPT：10 分钟答辩展⽰
五、评分标准
总分 100 分，各维度权重如下：
• 测评指标（30%）：CALVIN ABC→D 平均任务链⻓度，Task 1~5 各阶段成功率
• 技术⽅案（25%）：模型选择合理性、训练策略设计、创新性（新架构实现额外加分）
• Failure 分析（20%）：失败模式分类完整性、根因分析深度、改进⽅案有效性
• ⼯程实现（15%）：代码质量、可复现性、⽂档完整度
• 答辩表现（10%）：表达清晰度、技术深度、问答环节表现
六、参考资源
• StarVLA 仓库：github.com/starVLA/starVLA（建议使⽤ starVLA_dev 分⽀）
• StarVLA report：arXiv:2604.05014
• 预训练权重：HuggingFace StarVLA（含 Qwen2.5-VL-3B-Action、Qwen3-VL-4B-Action 等）
• CALVIN 基准：calvin.cs.uni-freiburg.de
• CosmoPredict2 ⽂档：StarVLA 仓库 docs/WM4A.md
• CALVIN 评测指南：StarVLA 仓库 examples/calvin/
七、注意事项
• ⿎励团队协作，但每⼈需独⽴完成技术⽂档中的个⼈贡献说明部分。
• 选择 VLA 或 WM 路线均可，但需在技术⽂档中详细论证选择原因，并分析另⼀路线的优劣。