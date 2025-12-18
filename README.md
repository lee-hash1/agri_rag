TARAG: A Time-aware Retrieval-augmented Generation Framework for Precise Agricultural Practice Support

[//]: # (https://img.shields.io/badge/Paper-arXiv-brightgreen)

[//]: # (https://img.shields.io/badge/License-MIT-blue)

[//]: # (https://img.shields.io/badge/Python-3.9%252B-blue)

[//]: # (https://img.shields.io/badge/PyTorch-2.5.1%252B-red)

📌 Project Overview
TARAG (Time-aware Retrieval-augmented Generation) is a framework designed to provide time-sensitive decision support for agricultural practices. By integrating time-aware knowledge base construction, hybrid time retrieval, and time-based generation, TARAG delivers precise, stage-accurate recommendations for pest and disease management aligned with crop phenology and seasonal constraints.

✨ Key Features
✅ Time-Aware Knowledge Base Construction: Extract and annotate time metadata from unstructured agricultural documents

✅ Hybrid Time Retrieval: Combine sparse retrieval (BM25) with dense semantic retrieval and time-sensitive re-ranking

✅ Time-Aware Generation: Generate responses conditioned on both semantic relevance and temporal alignment

✅ TAQA Dataset: First bilingual (Chinese-English) time-annotated agricultural QA dataset covering 2,000+ pests and diseases

✅ State-of-the-Art Performance: 99.14% retrieval recall and 66.85% F1 score for generated suggestions

📂 Project Structure
text
TARAG/
├── data/                    # Datasets and knowledge base
│   ├── TAQA/               # TAQA dataset (bilingual time-annotated QA pairs)
│   └── knowledge_base/     # Structured time-indexed knowledge base
├── modules/                # Core framework modules
│   ├── time_aware_kb/      # Time-aware knowledge base construction
│   ├── hybrid_retrieval/   # Hybrid time retrieval module
│   └── time_aware_gen/     # Time-aware answer generation
├── experiments/            # Experimental scripts and results
├── models/                 # Pretrained models and embeddings
├── utils/                  # Utility functions
├── configs/                # Configuration files
├── requirements.txt        # Python dependencies
├── train.py                # Training scripts
├── inference.py            # Inference scripts
├── evaluate.py             # Evaluation scripts
└── README.md               # This file
📊 TAQA Dataset
TAQA is a bilingual (Chinese-English) time-annotated agricultural question-answering dataset with approximately 30,000 high-quality QA pairs covering over 2,000 types of pests and diseases.

Time Annotation Categories:
Phenological Stage (e.g., "seedling stage", "flowering period")

Seasonal Period (e.g., "early spring", "late autumn")

Calendar-Based Time (e.g., "August", "late July")

Relative Time Expression (e.g., "before harvest", "after flowering")

Dataset Statistics:
Total QA pairs: ~30,000

Languages: Chinese and English

Time categories: 4 distinct types

Coverage: 2,000+ pests and diseases

Download and usage instructions: data/TAQA/README.md

🚀 Quick Start
1. Installation
bash
git clone https://github.com/your-username/TARAG.git
cd TARAG
pip install -r requirements.txt
2. Data Preparation
Download and extract the TAQA dataset and knowledge base to the data/ directory.

3. Basic Usage
python
from modules.hybrid_retrieval import HybridTimeRetriever
from modules.time_aware_gen import TimeAwareGenerator

# Initialize retriever and generator
retriever = HybridTimeRetriever(knowledge_base_path="data/knowledge_base/")
generator = TimeAwareGenerator(model_name="DeepSeek-R1-14B")

# Input query with time context
query = "How to control rice planthopper during seedling stage?"
time_context = "seedling stage"

# Retrieve and generate
documents = retriever.retrieve(query, time_context)
answer = generator.generate(query, documents, time_context)
print(answer)
4. Build Custom Knowledge Base
bash
python train.py --config configs/kb_build.yaml
📈 Experimental Results
Retrieval Performance (Recall@20)
Model	Recall@20
Contriever	22.62%
E5	97.51%
Qwen3-Embedding-4B	96.21%
TARAG (Ours)	99.14%
Generation Performance (F1@10 with DeepSeek-R1-14B)
Method	F1@10
Direct Prompt	9.33%
Naive RAG	51.22%
TimeR4	62.28%
TARAG	66.85%
Detailed results are available in the paper and experiments/ directory.

🧪 Citation
If you use TARAG or the TAQA dataset in your research, please cite:

bibtex
@article{liu2025tarag,
  title={TARAG: A Time-aware Retrieval-augmented Generation Framework for Precise Agricultural Practice Support},
  author={Liu, Lei and Li, Shunbao and Qi, Jun and Yuan, Zhipeng and Yang, Po},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2025}
}
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙌 Acknowledgements
We thank all contributors who provided data, models, and experimental support for this project. Special thanks to the TAQA dataset annotation team.