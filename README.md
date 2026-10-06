# AntiStrat-AI

Tactical analytics and scouting engine for Counter-Strike 2 based on spatio-temporal telemetry and deep representation learning.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Features

- **Telemetry Ingestion**: Extracts and cleans player trajectories from Source 2 binary demos (`.dem`) into structured CSV files.
- **Spatio-Temporal Normalization**: Normalizes continuous 3D coordinates, view angles, and player status across official competitive maps.
- **Deep Representation Learning**: PyTorch Bidirectional GRU Autoencoder compressing 5v5 team movements into compact 64-dimensional tactical embeddings.
- **Automated Playbook Clustering**: Unsupervised $K$-Means clustering grouping rounds into strategic archetypes (rushes, splits, map control) with automated scouting reports.
- **Vector Similarity Search**: In-memory cosine similarity retrieval engine to find historically analogous rounds for any given situation.

---

## Installation

```bash
git clone https://github.com/delaubier/AntiStrat-AI.git
cd AntiStrat-AI

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## Usage

### 1. Run the Full Pipeline

Executes the complete pipeline end-to-end (ingestion, dataset building, autoencoder training, clustering, and vector search):

```bash
python run_pipeline.py
```

### 2. Python API Example

Each module can be used independently in custom workflows:

```python
from src.parser.demo_extractor import CS2DemoExtractor
from src.models.dataset import CS2TacticalDataset
from src.models.trainer import TacticalTrainer
from src.clustering.vector_index import TacticalVectorIndex

# 1. Ingest demo and export trajectories to CSV
extractor = CS2DemoExtractor("data/raw_demos/match.dem")
csv_path = extractor.process_and_save()

# 2. Build PyTorch dataset and extract tactical embeddings
dataset = CS2TacticalDataset(csv_path, team_side="TERRORIST")
trainer = TacticalTrainer()
embeddings_df = trainer.extract_embeddings(dataset)

# 3. Find top 3 most similar rounds to Round 3
index = TacticalVectorIndex(embeddings_df)
results = index.search_similar(query_round_index=3, top_k=3)

for rank, item in enumerate(results, 1):
    print(f"Top {rank}: Round {item['round_index']} (Similarity: {item['similarity_score']}%)")
```

---

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.