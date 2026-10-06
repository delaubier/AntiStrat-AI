"""
Simple Map Normalizer for CS2.
Scales player coordinates (X, Y, Z), yaw and health between 0.0 and 1.0.
"""

import numpy as np
import pandas as pd


# Bornes réelles des maps officielles CS2 (bounding boxes de référence)
MAP_BOUNDS = {
    "de_dust2": {
        "min_x": -2480.0, "max_x": 2120.0,
        "min_y": -1250.0, "max_y": 3400.0,
        "min_z": -250.0,  "max_z": 450.0
    },
    "de_mirage": {
        "min_x": -3230.0, "max_x": 1900.0,
        "min_y": -3400.0, "max_y": 1700.0,
        "min_z": -400.0,  "max_z": 400.0
    },
    "de_inferno": {
        "min_x": -2100.0, "max_x": 2900.0,
        "min_y": -1200.0, "max_y": 3900.0,
        "min_z": -200.0,  "max_z": 500.0
    },
    "de_nuke": {
        "min_x": -3450.0, "max_x": 3750.0,
        "min_y": -4300.0, "max_y": 2900.0,
        "min_z": -800.0,  "max_z": 500.0
    },
    "de_ancient": {
        "min_x": -2950.0, "max_x": 2150.0,
        "min_y": -2900.0, "max_y": 2200.0,
        "min_z": -300.0,  "max_z": 400.0
    },
    "de_anubis": {
        "min_x": -2800.0, "max_x": 2400.0,
        "min_y": -2000.0, "max_y": 3300.0,
        "min_z": -200.0,  "max_z": 400.0
    }
}


class MapNormalizer:
    def __init__(self, bounds_dict: dict = None):
        self.bounds = bounds_dict or MAP_BOUNDS

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajoute les colonnes normalisées (X_norm, Y_norm, Z_norm, yaw_norm, health_norm)
        toutes comprises entre 0.0 et 1.0.
        """
        df = df.copy()
        map_name = df["map_name"].iloc[0]

        if map_name not in self.bounds:
            raise ValueError(f"Map '{map_name}' inconnue. Maps supportées : {list(self.bounds.keys())}")

        b = self.bounds[map_name]

        # Normalisation Min-Max ramenée entre 0.0 et 1.0
        df["X_norm"] = np.clip((df["X"] - b["min_x"]) / (b["max_x"] - b["min_x"]), 0.0, 1.0)
        df["Y_norm"] = np.clip((df["Y"] - b["min_y"]) / (b["max_y"] - b["min_y"]), 0.0, 1.0)
        df["Z_norm"] = np.clip((df["Z"] - b["min_z"]) / (b["max_z"] - b["min_z"]), 0.0, 1.0)

        # Normalisation de la vue (-180° à +180° -> 0.0 à 1.0) et de la vie (0 à 100 -> 0.0 à 1.0)
        df["yaw_norm"] = np.clip((df["yaw"] + 180.0) / 360.0, 0.0, 1.0)
        df["health_norm"] = np.clip(df["health"] / 100.0, 0.0, 1.0)

        return df

    def denormalize(self, x_norm: float, y_norm: float, map_name: str):
        """Convertit des coordonnées normalisées (0-1) vers les coordonnées de jeu réelles."""
        b = self.bounds[map_name]
        x_real = x_norm * (b["max_x"] - b["min_x"]) + b["min_x"]
        y_real = y_norm * (b["max_y"] - b["min_y"]) + b["min_y"]
        return x_real, y_real


if __name__ == "__main__":
    # Test direct sur les données extraites
    data_path = "data/processed/100-thieves-vs-heroic-m1-dust2_trajectories.csv"
    print(f"[*] Chargement de {data_path}...")
    
    df = pd.read_csv(data_path)
    normalizer = MapNormalizer()
    normalized_df = normalizer.normalize(df)

    print("\n[+] Aperçu des valeurs normalisées (toutes entre 0.0 et 1.0) :")
    cols = ["X_norm", "Y_norm", "Z_norm", "yaw_norm", "health_norm"]
    print(normalized_df[cols].describe().round(3))
