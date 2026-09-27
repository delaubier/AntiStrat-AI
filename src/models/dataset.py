"""
PyTorch Dataset and DataLoader for CS2 Tactical Trajectories.
Converts normalized round trajectory DataFrames into fixed-size tensors (Timesteps x Features).
"""

import sys
from pathlib import Path

# Assure que la racine du projet est dans sys.path quel que soit le dossier d'exécution
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from typing import Union, List, Dict, Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.parser.map_normalizer import MapNormalizer


class CS2TacticalDataset(Dataset):
    """
    Transforms CS2 round trajectories into PyTorch tensors for tactical deep learning.
    
    Each sample represents 1 round for 1 team (e.g. TERRORIST).
    Shape per sample: (T_timesteps=46, 5_players * 6_features = 30)
    """

    FEATURE_COLS = ["X_norm", "Y_norm", "Z_norm", "yaw_norm", "health_norm", "is_alive"]

    def __init__(
        self,
        data_source: Union[str, Path, pd.DataFrame],
        team_side: str = "TERRORIST",
        expected_players: int = 5,
        expected_timesteps: int = 46,
        normalizer: MapNormalizer = None
    ):
        """
        :param data_source: Chemin vers le fichier .parquet ou DataFrame Pandas
        :param team_side: 'TERRORIST' (attaque) ou 'CT' (défense)
        :param expected_players: Nombre de joueurs attendus par équipe (défaut: 5)
        :param expected_timesteps: Nombre de secondes analysées (défaut: 46 pour 0s à 45s)
        :param normalizer: Instance de MapNormalizer (optionnel)
        """
        self.team_side = team_side.upper()
        self.expected_players = expected_players
        self.expected_timesteps = expected_timesteps
        self.features_per_player = len(self.FEATURE_COLS)
        self.total_features = self.expected_players * self.features_per_player

        # 1. Chargement des données
        if isinstance(data_source, (str, Path)):
            df = pd.read_parquet(data_source)
        elif isinstance(data_source, pd.DataFrame):
            df = data_source.copy()
        else:
            raise TypeError("data_source doit être un chemin (str/Path) ou un pd.DataFrame")

        # 2. Normalisation si les colonnes normalisées ne sont pas encore présentes
        if "X_norm" not in df.columns:
            norm = normalizer or MapNormalizer()
            df = norm.normalize(df)

        # 3. Filtrage sur le camp choisi
        self.df = df[df["team_name"] == self.team_side].copy()
        self.df["is_alive"] = self.df["is_alive"].astype(float)

        # 4. Construction des tenseurs
        self.samples: List[torch.Tensor] = []
        self.metadata: List[Dict[str, Any]] = []
        self._build_tensors()

    def _build_tensors(self):
        """Construit les tenseurs (46, 30) pour chaque round valide."""
        grouped_rounds = self.df.groupby("round_index")

        for round_idx, round_df in grouped_rounds:
            players = sorted(round_df["steamid"].unique())

            # Vérification du nombre de joueurs
            if len(players) != self.expected_players:
                # Round incomplet (déconnexion, warmup), on l'écarte pour la qualité des données
                continue

            # Matrice pour ce round : (46 secondes, 5 joueurs * 6 features = 30)
            round_matrix = np.zeros((self.expected_timesteps, self.total_features), dtype=np.float32)
            is_valid = True

            for p_idx, steam_id in enumerate(players):
                player_df = round_df[round_df["steamid"] == steam_id].sort_values("round_seconds")
                
                # Vérifier qu'on a bien les 46 pas de temps
                if len(player_df) != self.expected_timesteps:
                    is_valid = False
                    break

                # Extraction des 6 features numériques
                player_feats = player_df[self.FEATURE_COLS].to_numpy(dtype=np.float32)

                # Remplissage du bloc pour ce joueur : colonnes [p*6 : (p+1)*6]
                start_col = p_idx * self.features_per_player
                end_col = start_col + self.features_per_player
                round_matrix[:, start_col:end_col] = player_feats

            if is_valid:
                self.samples.append(torch.from_numpy(round_matrix))
                self.metadata.append({
                    "round_index": int(round_idx),
                    "team_side": self.team_side,
                    "map_name": round_df["map_name"].iloc[0] if "map_name" in round_df.columns else "unknown",
                    "demo_name": round_df["demo_name"].iloc[0] if "demo_name" in round_df.columns else "unknown"
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Renvoie le tenseur du round demandé (forme: (46, 30))."""
        return self.samples[idx]

    def get_metadata(self, idx: int) -> Dict[str, Any]:
        """Renvoie les métadonnées associées au round (numéro, map, équipe)."""
        return self.metadata[idx]


def get_dataloader(
    parquet_path: Union[str, Path],
    team_side: str = "TERRORIST",
    batch_size: int = 4,
    shuffle: bool = True
) -> DataLoader:
    """
    Fonction utilitaire pour obtenir directement un DataLoader prêt à l'emploi.
    """
    dataset = CS2TacticalDataset(parquet_path, team_side=team_side)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":
    data_file = "data/processed/100-thieves-vs-heroic-m1-dust2_trajectories.parquet"
    print(f"[*] Chargement du Dataset pour l'équipe attaquante (TERRORIST)...")
    
    dataset_t = CS2TacticalDataset(data_file, team_side="TERRORIST")
    print(f"[+] Nombre de rounds T valides : {len(dataset_t)}")
    
    if len(dataset_t) > 0:
        sample_tensor = dataset_t[0]
        print(f"[+] Forme d'un round unique (Tenseur PyTorch) : {sample_tensor.shape}")
        print(f"    -> {sample_tensor.shape[0]} pas de temps (secondes 0 à 45)")
        print(f"    -> {sample_tensor.shape[1]} features (5 joueurs * 6 features : X, Y, Z, yaw, health, is_alive)")

        # Test du DataLoader
        print("\n[*] Test du DataLoader (batch_size=4, shuffle=True) :")
        loader = DataLoader(dataset_t, batch_size=4, shuffle=True)
        for batch_idx, batch_tensors in enumerate(loader):
            print(f"    Batch #{batch_idx + 1} shape : {batch_tensors.shape} (Type: {batch_tensors.dtype})")
            break
