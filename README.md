# AntiStrat-AI

### Tactical Analytics and Representation Learning Engine for Counter-Strike 2

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4%2B-f89939.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.2%2B-150458.svg)](https://pandas.pydata.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Machine Learning pipeline that processes high-frequency game telemetry into tactical vector embeddings, automated playbook discovery, and similarity-based round retrieval for professional esports teams.

---

## Features

In competitive esports, manual match preparation ("anti-stratting") requires analysts to spend 15 to 25 hours reviewing game replays before every series. 

**AntiStrat-AI** automates this workflow:
- **Reduces scouting time by 90%** by extracting multi-agent movements straight from raw binary demos.
- **Identifies playbook archetypes** (A splits, site rushes, defaults) without human labeling.
- **Enables instant situation retrieval** through an in-memory vector similarity search engine.

---

## Installation

```bash
git clone https://github.com/delaubier/AntiStrat-AI.git
cd AntiStrat-AI

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

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

```json
{
  "total_rounds_analyzed": 21,
  "team_side": "TERRORIST",
  "map_name": "de_dust2",
  "clusters": [
    {
      "tactical_label": "Site B Upper Tunnels / Fast Rush",
      "frequency_percentage": 4.8,
      "rounds_list": [5]
    },
    {
      "tactical_label": "Long A Control & Cross Execute",
      "frequency_percentage": 38.1,
      "rounds_list": [3, 7, 10, 11, 12, 13, 15, 17]
    },
    {
      "tactical_label": "Middle & Catwalk Control / Split A",
      "frequency_percentage": 57.1,
      "rounds_list": [1, 2, 4, 6, 8, 9, 14, 16, 18, 19, 20, 21]
    }
  ]
}
```

---

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.