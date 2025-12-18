# TARAG: A Time-aware Retrieval-augmented Generation Framework for Precise Agricultural Practice Support

[//]: # ([![Paper]&#40;https://img.shields.io/badge/Paper-arXiv-brightgreen&#41;]&#40;https://arxiv.org/abs/你的论文链接&#41;)

[//]: # ([![License]&#40;https://img.shields.io/badge/License-MIT-blue&#41;]&#40;LICENSE&#41;)

[//]: # ([![Python]&#40;https://img.shields.io/badge/Python-3.9%2B-blue&#41;]&#40;https://www.python.org/&#41;)

## 📌 项目简介

TARAG（Time-aware Retrieval-augmented Generation）是一个面向精准农业决策支持的时间感知检索增强生成框架。该框架结合时间感知的知识库构建、混合时间检索与时间感知生成三大模块，旨在为病虫害防治等农业场景提供符合作物物候期、季节变化的时间敏感建议。

## ✨ 主要特点

- ✅ **时间感知知识库构建**：从非结构化文本中提取并标注时间元数据，构建结构化的农业知识库
- ✅ **混合时间检索**：结合稀疏检索（BM25）与密集语义检索，并引入时间重排序机制
- ✅ **时间感知生成**：在生成阶段引入时间约束提示，确保推荐内容的时间一致性
- ✅ **TAQA数据集**：首个大规模中英双语时间标注农业问答数据集，覆盖2000+病虫害类型
- ✅ **高性能检索与生成**：在多项评测中显著优于现有RAG框架，检索召回率达99.14%，生成F1达66.85%

## 📂 项目结构
TARAG/
├── data/ # 数据集与知识库文件
│ ├── TAQA/ # TAQA数据集（双语时间标注QA对）
│ └── knowledge_base/ # 结构化时间索引知识库
├── modules/ # 核心模块
│ ├── time_aware_kb/ # 时间感知知识库构建
│ ├── hybrid_retrieval/ # 混合时间检索模块
│ └── time_aware_gen/ # 时间感知生成模块
├── experiments/ # 实验脚本与评估结果
├── models/ # 预训练模型与嵌入文件
├── utils/ # 工具函数
├── configs/ # 配置文件
├── requirements.txt # 依赖包列表
├── train.py # 训练脚本
├── inference.py # 推理脚本
└── README.md # 项目说明

text

## 📊 TAQA 数据集

TAQA 是一个中英双语、时间标注的农业问答数据集，包含约3万条高质量QA对，覆盖超过2000种病虫害，并标注了四类时间表达：

- **物候期**（如“苗期”、“开花期”）
- **季节性时段**（如“早春”、“晚秋”）
- **日历时间**（如“八月”、“九月下旬”）
- **相对时间表达**（如“收获前”、“开花后”）

数据集下载与使用说明详见 [`data/TAQA/README.md`](data/TAQA/README.md)。

## 🚀 快速开始

### 1. 环境安装

```bash
git clone https://github.com/your-username/TARAG.git
cd TARAG
pip install -r requirements.txt
2. 数据准备
下载并解压TAQA数据集与预训练知识库至 data/ 目录。

3. 运行推理示例
python
from modules.hybrid_retrieval import HybridTimeRetriever
from modules.time_aware_gen import TimeAwareGenerator

# 初始化检索器与生成器
retriever = HybridTimeRetriever(knowledge_base_path="data/knowledge_base/")
generator = TimeAwareGenerator(model_name="DeepSeek-R1-14B")

# 输入查询
query = "水稻苗期如何防治稻飞虱？"
time_context = "苗期"

# 检索与生成
documents = retriever.retrieve(query, time_context)
answer = generator.generate(query, documents, time_context)
print(answer)
4. 训练自定义知识库
bash
python train.py --config configs/kb_build.yaml
📈 实验结果
检索性能对比（Recall@20）
模型	Recall@20
Contriever	22.62%
E5	97.51%
Qwen3-Embedding-4B	96.21%
TARAG (Ours)	99.14%
生成性能对比（F1@10）
模型 + 方法	F1@10
DeepSeek-R1-14B + Naive RAG	51.22%
DeepSeek-R1-14B + TimeR4	62.28%
DeepSeek-R1-14B + TARAG	66.85%
更多详细结果请见论文与 experiments/ 目录。

🧪 引用
如果本项目或论文对您的研究有帮助，请引用：

bibtex
@article{liu2025tarag,
  title={TARAG: A Time-aware Retrieval-augmented Generation Framework for Precise Agricultural Practice Support},
  author={Liu, Lei and Li, Shunbao and Qi, Jun and Yuan, Zhipeng and Yang, Po},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2025}
}
📄 许可证
本项目采用 MIT 许可证。详见 LICENSE 文件。

🙌 致谢
感谢所有为本项目提供数据、模型与实验支持的机构与个人。特别感谢TAQA数据集的标注团队。