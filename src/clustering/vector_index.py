"""
Tactical Vector Index for CS2.
Provides fast similarity search to find similar rounds based on tactical embeddings.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from typing import List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class TacticalVectorIndex:
    """
    In-memory vector search engine for CS2 tactical embeddings.
    Allows coaches to find the most similar rounds to a given reference round.
    """

    def __init__(self, embeddings_df: pd.DataFrame = None):
        self.round_indices = []
        self.metadata = []
        self.embeddings_matrix = None

        if embeddings_df is not None:
            self.build_index(embeddings_df)

    def build_index(self, embeddings_df: pd.DataFrame):
        """Construit la matrice vectorielle en mémoire."""
        self.round_indices = embeddings_df["round_index"].tolist()
        self.metadata = embeddings_df[["round_index", "team_side", "map_name", "demo_name"]].to_dict(orient="records")
        
        # Matrice (N_rounds, 64)
        self.embeddings_matrix = np.stack(embeddings_df["embedding"].values)

    def search_similar(self, query_round_index: int, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Trouve les top_k rounds les plus similaires au round demandé.
        """
        if query_round_index not in self.round_indices:
            raise ValueError(f"Round {query_round_index} introuvable dans l'index.")

        target_idx = self.round_indices.index(query_round_index)
        target_vector = self.embeddings_matrix[target_idx].reshape(1, -1)

        # Calcul de la similarité cosinus avec tous les autres rounds
        similarities = cosine_similarity(target_vector, self.embeddings_matrix)[0]

        # Tri par score décroissant (en excluant le round lui-même)
        sorted_indices = np.argsort(similarities)[::-1]
        
        results = []
        for idx in sorted_indices:
            r_id = self.round_indices[idx]
            if r_id == query_round_index:
                continue  # Ne pas se comparer à soi-même

            score = float(similarities[idx])
            res = {
                "round_index": r_id,
                "similarity_score": round(score * 100, 2),  # En pourcentage (ex: 95.4%)
                "metadata": self.metadata[idx]
            }
            results.append(res)
            if len(results) >= top_k:
                break

        return results


if __name__ == "__main__":
    import torch
    from src.models.dataset import CS2TacticalDataset
    from src.models.trainer import TacticalTrainer

    data_file = "data/processed/100-thieves-vs-heroic-m1-dust2_trajectories.csv"
    dataset = CS2TacticalDataset(data_file, team_side="TERRORIST")
    
    trainer = TacticalTrainer()
    weights_path = Path("data/embeddings/tactical_autoencoder.pt")
    if weights_path.exists():
        trainer.model.load_state_dict(torch.load(weights_path))
    
    emb_df = trainer.extract_embeddings(dataset)

    # Création de l'index vectoriel
    index = TacticalVectorIndex(emb_df)
    
    # Test : Recherche de similarité pour le Round 3
    test_round = 3
    print(f"[*] Recherche des rounds les plus similaires au Round #{test_round}...")
    similar_rounds = index.search_similar(query_round_index=test_round, top_k=3)

    print(f"\n[TOP 3 ROUNDS LES PLUS PROCHES DU ROUND #{test_round}] :")
    for rank, res in enumerate(similar_rounds, 1):
        print(f"  #{rank} : Round {res['round_index']} (Similarite : {res['similarity_score']}%)")
