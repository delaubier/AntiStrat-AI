"""
Spatio-Temporal Autoencoder for CS2 Tactical Embeddings.
Compresses 5v5 player trajectory sequences into dense vector representations.
"""

import sys
from pathlib import Path

# Assure que la racine du projet est dans sys.path quel que soit le dossier d'exécution
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
import torch.nn as nn
import torch.nn.functional as F


class TacticalEncoder(nn.Module):
    """
    Encodeur séquentiel (Bidirectional GRU) qui compresse une séquence de round (46s x 30 features)
    en un vecteur d'embedding compact (ex: 64 dimensions).
    """

    def __init__(
        self,
        input_dim: int = 30,
        hidden_dim: int = 128,
        embedding_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        # Comme le GRU est bidirectionnel, la dimension cachée est 2 * hidden_dim
        self.fc = nn.Linear(hidden_dim * 2, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Tenseur de forme (batch_size, timesteps=46, features=30)
        :return: Embedding normalisé L2 de forme (batch_size, embedding_dim=64)
        """
        out, _ = self.gru(x)  # out: (batch_size, timesteps, hidden_dim * 2)
        
        # On utilise le pooling moyen temporel pour résumer l'ensemble du round
        pooled = torch.mean(out, dim=1)  # (batch_size, hidden_dim * 2)
        
        embedding = self.fc(pooled)      # (batch_size, embedding_dim)
        
        # Normalisation L2 (rend les embeddings comparables via similarité cosinus)
        normalized_embedding = F.normalize(embedding, p=2, dim=-1)
        return normalized_embedding


class TacticalDecoder(nn.Module):
    """
    Décodeur séquentiel qui tente de reconstruire la trajectoire originale (46s x 30 features)
    à partir du seul vecteur d'embedding.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 30,
        timesteps: int = 46,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.timesteps = timesteps
        self.embedding_dim = embedding_dim

        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        :param embedding: Vecteur de forme (batch_size, embedding_dim)
        :return: Séquence reconstruite de forme (batch_size, timesteps=46, output_dim=30)
        """
        batch_size = embedding.size(0)
        
        # On répète l'embedding sur les 46 pas de temps pour alimenter le GRU
        repeated_emb = embedding.unsqueeze(1).repeat(1, self.timesteps, 1)  # (B, 46, embedding_dim)
        
        out, _ = self.gru(repeated_emb)   # (B, 46, hidden_dim)
        recon = self.fc_out(out)           # (B, 46, output_dim)
        
        # Sigmoid pour contraindre la reconstruction entre 0.0 et 1.0 (nos données normalisées)
        return torch.sigmoid(recon)


class CS2TacticalAutoencoder(nn.Module):
    """
    Autoencodeur Tactique complet combinant l'Encodeur et le Décodeur.
    """

    def __init__(
        self,
        input_dim: int = 30,
        hidden_dim: int = 128,
        embedding_dim: int = 64,
        timesteps: int = 46,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.encoder = TacticalEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        self.decoder = TacticalDecoder(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            timesteps=timesteps,
            num_layers=num_layers,
            dropout=dropout
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extrait l'embedding tactique (à utiliser pour le clustering et le scouting)."""
        return self.encoder(x)

    def decode(self, embedding: torch.Tensor) -> torch.Tensor:
        """Reconstruit la séquence à partir de l'embedding."""
        return self.decoder(embedding)

    def forward(self, x: torch.Tensor):
        """Passe avant complète pendant l'apprentissage."""
        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)
        return reconstruction, embedding


if __name__ == "__main__":
    from src.models.dataset import get_dataloader

    print("[*] Initialisation de l'Autoencodeur CS2...")
    model = CS2TacticalAutoencoder(input_dim=30, hidden_dim=128, embedding_dim=64, timesteps=46)
    
    # Nombre de paramètres
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[+] Modèle créé avec succès ! ({total_params:,} paramètres entraînables)")

    # Test avec de vraies données du DataLoader
    data_file = "data/processed/100-thieves-vs-heroic-m1-dust2_trajectories.parquet"
    loader = get_dataloader(data_file, team_side="TERRORIST", batch_size=4)

    for batch in loader:
        print(f"\n[*] Entrée batch réelle : {batch.shape}")
        
        # Test de l'encodage (la partie qui nous intéresse pour le scouting)
        embeddings = model.encode(batch)
        print(f"[+] Embeddings générés : {embeddings.shape} (Vecteurs de dimension 64 normalisés L2)")
        
        # Test de la reconstruction complète
        recon, emb = model(batch)
        print(f"[+] Reconstruction sortie : {recon.shape}")
        
        # Calcul de la perte de reconstruction (MSE)
        loss = nn.MSELoss()(recon, batch)
        print(f"[+] Perte MSE initiale (non entraîné) : {loss.item():.4f}")
        break
