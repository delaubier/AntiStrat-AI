# AntiStrat-AI

### Moteur d'analyse tactique et d'apprentissage de représentations pour Counter-Strike 2

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4%2B-f89939.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.2%2B-150458.svg)](https://pandas.pydata.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

🌐 **Langue / Language** : [English 🇬🇧](README.md) • **Français 🇫🇷**

Un pipeline complet de Machine Learning qui transforme la télémétrie de jeu à haute fréquence en plongements vectoriels (*embeddings*) tactiques, automatise la détection de schémas de jeu (*playbooks*) et permet la recherche de manches par similarité pour les équipes esport professionnelles.

---

## Fonctionnalités

Sur la scène compétitive esport, la préparation manuelle des matchs (« anti-stratting ») oblige les analystes à passer entre 15 et 25 heures à visionner des démos avant chaque série. 

**AntiStrat-AI** automatise ce flux de travail :
- **Réduit le temps de scouting de 90 %** en extrayant les trajectoires multi-agents directement depuis les fichiers démo binaires bruts.
- **Identifie les archétypes de jeu** (*splits* A, *rushes* de site, configurations par défaut) sans étiquetage manuel préalable.
- **Permet une recherche instantanée de situations** grâce à un moteur de similarité vectorielle en mémoire.

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

## Utilisation

### 1. Exécuter l'ensemble du pipeline

Lance le pipeline complet de bout en bout (ingestion, constitution du jeu de données, entraînement de l'auto-encodeur, partitionnement/clustering et recherche vectorielle) :

```bash
python run_pipeline.py
```

### 2. Exemple de sortie / API Python

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

## Licence

Distribué sous la licence MIT. Voir [LICENSE](LICENSE) pour plus d'informations.
