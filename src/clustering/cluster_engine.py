"""
Tactical Clustering Engine for CS2.
Groups tactical embeddings into strategic playbooks (e.g., Rush B, Mid to Short, Long A)
and generates automated scouting reports for coaches.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class TacticalClusterEngine:
    """
    Groups round embeddings into tactical clusters and extracts scouting summaries.
    """

    def __init__(self, n_clusters: int = 3):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.cluster_labels = None
        self.cluster_centers = None

    def fit_predict(self, embeddings_df: pd.DataFrame) -> pd.DataFrame:
        """
        Applique le clustering K-Means sur les vecteurs d'embeddings.
        :param embeddings_df: DataFrame avec colonnes ['round_index', 'embedding', ...]
        :return: DataFrame avec la colonne 'cluster_id' ajoutée.
        """
        # Matrice numpy des embeddings : (N_rounds, 64)
        X = np.stack(embeddings_df["embedding"].values)

        # Ajustement du nombre de clusters si l'échantillon est trop petit
        actual_k = min(self.n_clusters, len(X))
        if actual_k < 2:
            df_out = embeddings_df.copy()
            df_out["cluster_id"] = 0
            return df_out

        self.kmeans = KMeans(n_clusters=actual_k, random_state=42, n_init=10)
        self.cluster_labels = self.kmeans.fit_predict(X)
        self.cluster_centers = self.kmeans.cluster_centers_

        df_out = embeddings_df.copy()
        df_out["cluster_id"] = self.cluster_labels
        return df_out

    def generate_scouting_report(
        self,
        clustered_df: pd.DataFrame,
        trajectories_df: pd.DataFrame = None
    ) -> Dict[str, Any]:
        """
        Génère un rapport de scouting détaillé lisible par un coach.
        """
        total_rounds = len(clustered_df)
        report = {
            "total_rounds_analyzed": total_rounds,
            "team_side": clustered_df["team_side"].iloc[0] if not clustered_df.empty else "N/A",
            "map_name": clustered_df["map_name"].iloc[0] if not clustered_df.empty else "N/A",
            "clusters": []
        }

        for c_id in sorted(clustered_df["cluster_id"].unique()):
            sub = clustered_df[clustered_df["cluster_id"] == c_id]
            rounds_list = sub["round_index"].tolist()
            count = len(sub)
            freq = round((count / total_rounds) * 100, 1)

            # Analyse spatiale si les trajectoires brutes sont fournies
            strat_label = f"Tactique #{c_id + 1}"
            if trajectories_df is not None and not trajectories_df.empty:
                # Regarder la position moyenne de l'équipe à 20s
                c_trajs = trajectories_df[
                    (trajectories_df["round_index"].isin(rounds_list)) &
                    (trajectories_df["round_seconds"] == 20.0)
                ]
                if not c_trajs.empty:
                    mean_x = c_trajs["X_norm"].mean()
                    mean_y = c_trajs["Y_norm"].mean()
                    
                    # Déduction de la zone sur Dust2
                    if mean_x < 0.35 and mean_y > 0.5:
                        strat_label = "Attaque Site B (Upper Tunnels / Rush B)"
                    elif mean_x > 0.6:
                        strat_label = "Contrôle Long A / Sortie A"
                    else:
                        strat_label = "Contrôle Middle & Catwalk / Split"

            cluster_info = {
                "cluster_id": int(c_id),
                "tactical_label": strat_label,
                "rounds_count": count,
                "frequency_percentage": freq,
                "rounds_list": rounds_list
            }
            report["clusters"].append(cluster_info)

        return report


if __name__ == "__main__":
    from src.models.dataset import CS2TacticalDataset
    from src.models.trainer import TacticalTrainer

    data_file = "data/processed/100-thieves-vs-heroic-m1-dust2_trajectories.parquet"
    dataset = CS2TacticalDataset(data_file, team_side="TERRORIST")
    
    # 1. Extraction des embeddings avec le modèle entraîné
    trainer = TacticalTrainer()
    weights_path = Path("data/embeddings/tactical_autoencoder.pt")
    if weights_path.exists():
        trainer.model.load_state_dict(torch.load(weights_path))
    
    emb_df = trainer.extract_embeddings(dataset)

    # 2. Clustering tactique (ex: 3 groupes de stratégies)
    print("\n[*] Analyse et regroupement des tactiques (Clustering)...")
    engine = TacticalClusterEngine(n_clusters=3)
    clustered_df = engine.fit_predict(emb_df)

    # 3. Génération du rapport de scouting
    raw_trajectories = pd.read_parquet(data_file)
    from src.parser.map_normalizer import MapNormalizer
    raw_trajectories = MapNormalizer().normalize(raw_trajectories)

    report = engine.generate_scouting_report(clustered_df, raw_trajectories)
    print("\n" + "="*60)
    print(f"[RAPPORT DE SCOUTING TACTIQUE] - {report['map_name'].upper()} ({report['team_side']})")
    print(f"Total des manches analysees : {report['total_rounds_analyzed']}")
    print("="*60)
    
    for c in report["clusters"]:
        print(f"\n> [{c['tactical_label']}]")
        print(f"  - Frequence : {c['frequency_percentage']}% ({c['rounds_count']} manches)")
        print(f"  - Manches concernees : {c['rounds_list']}")

    # Sauvegarde du rapport en JSON
    out_json = Path("data/embeddings/tactical_scouting_report.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Rapport exporté dans : {out_json}")
