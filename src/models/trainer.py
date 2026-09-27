"""
Trainer for CS2 Tactical Autoencoder.
Trains the spatio-temporal neural network to learn tactical representations and extracts embeddings.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.dataset import CS2TacticalDataset, get_dataloader
from src.models.tactical_encoder import CS2TacticalAutoencoder


class TacticalTrainer:
    """
    Manages the training loop of the CS2 Tactical Autoencoder and embedding extraction.
    """

    def __init__(
        self,
        model: CS2TacticalAutoencoder = None,
        learning_rate: float = 1e-3,
        device: str = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model or CS2TacticalAutoencoder(
            input_dim=30,
            hidden_dim=128,
            embedding_dim=64,
            timesteps=46
        )
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.criterion = nn.MSELoss()

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Entraîne le modèle sur une époque complète."""
        self.model.train()
        total_loss = 0.0

        for batch in dataloader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            reconstruction, _ = self.model(batch)
            loss = self.criterion(reconstruction, batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item() * batch.size(0)

        return total_loss / len(dataloader.dataset)

    def train(
        self,
        dataloader: DataLoader,
        epochs: int = 50,
        save_path: str = "data/embeddings/tactical_autoencoder.pt",
        verbose: bool = True
    ) -> list:
        """
        Boucle d'entraînement complète avec sauvegarde du meilleur modèle.
        """
        save_file = Path(save_path)
        save_file.parent.mkdir(parents=True, exist_ok=True)

        history = []
        best_loss = float("inf")

        if verbose:
            print(f"[*] Début de l'entraînement sur {self.device.upper()} ({len(dataloader.dataset)} rounds)...")

        for epoch in range(1, epochs + 1):
            loss = self.train_epoch(dataloader)
            history.append(loss)

            if loss < best_loss:
                best_loss = loss
                torch.save(self.model.state_dict(), save_file)

            if verbose and (epoch == 1 or epoch % 10 == 0 or epoch == epochs):
                print(f"    Epoch [{epoch:02d}/{epochs:02d}] - Perte MSE : {loss:.5f}")

        if verbose:
            print(f"[+] Entraînement terminé ! Meilleure perte : {best_loss:.5f}")
            print(f"[+] Poids sauvegardés dans : {save_file}")

        return history

    def extract_embeddings(self, dataset: CS2TacticalDataset) -> pd.DataFrame:
        """
        Fait passer tous les rounds du dataset dans l'encodeur
        et renvoie un DataFrame contenant les métadonnées et les vecteurs d'embeddings.
        """
        self.model.eval()
        results = []

        loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                embeddings = self.model.encode(batch).cpu().numpy()

                for idx in range(len(dataset)):
                    meta = dataset.get_metadata(idx)
                    row = {
                        "round_index": meta["round_index"],
                        "team_side": meta["team_side"],
                        "map_name": meta["map_name"],
                        "demo_name": meta["demo_name"],
                        "embedding": embeddings[idx]  # Array numpy (64,)
                    }
                    results.append(row)

        return pd.DataFrame(results)


if __name__ == "__main__":
    data_file = "data/processed/100-thieves-vs-heroic-m1-dust2_trajectories.parquet"
    
    # 1. Préparation du DataLoader
    dataset = CS2TacticalDataset(data_file, team_side="TERRORIST")
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 2. Entraînement de l'autoencodeur
    trainer = TacticalTrainer(learning_rate=2e-3)
    trainer.train(loader, epochs=40)

    # 3. Extraction des embeddings finaux
    print("\n[*] Extraction des embeddings tactiques de chaque round...")
    emb_df = trainer.extract_embeddings(dataset)
    print(f"[+] {len(emb_df)} rounds encodés en vecteurs de dimension {len(emb_df['embedding'].iloc[0])} !")
    print(emb_df[["round_index", "team_side", "map_name"]].head())
