"""
Demo Extractor for Counter-Strike 2 (.dem files)
Uses demoparser2 and Pandas to extract and clean
player spatio-temporal trajectories for tactical analysis.
"""

from pathlib import Path
from typing import Optional, List
import pandas as pd
from demoparser2 import DemoParser


class CS2DemoExtractor:
    """
    Extracts structured round trajectories and player movements from CS2 demos.
    """

    DEFAULT_FIELDS = [
        "X", "Y", "Z",             # Coordonnées 3D
        "pitch", "yaw",            # Orientation visée
        "team_name",               # 'TERRORIST' ou 'CT'
        "name",                    # Pseudo du joueur
        "steamid",                 # Identifiant unique Steam
        "is_alive",                # Statut de vie
        "health",                  # Points de vie (0-100)
        "total_rounds_played"      # Numéro de round
    ]

    def __init__(self, demo_path: str, tickrate: float = 64.0):
        self.demo_path = Path(demo_path)
        if not self.demo_path.exists():
            raise FileNotFoundError(f"Démo introuvable : {self.demo_path}")
        
        self.tickrate = tickrate
        self.parser = DemoParser(str(self.demo_path))

    def get_header_info(self) -> dict:
        """Récupère les métadonnées de base de la démo (map, serveur, etc.)."""
        return self.parser.parse_header()

    def get_round_intervals(self) -> pd.DataFrame:
        """
        Extrait les ticks de début de round (après le freeze time).
        round_freeze_end marque le moment précis où les joueurs peuvent bouger (T = 0s).
        """
        freeze_ends = self.parser.parse_event("round_freeze_end")
        if freeze_ends.empty:
            raise ValueError("Aucun événement 'round_freeze_end' trouvé dans la démo.")

        # On trie par tick et on assigne un numéro de manche séquentiel
        freeze_ends = freeze_ends.sort_values("tick").reset_index(drop=True)
        freeze_ends["round_index"] = freeze_ends.index + 1
        return freeze_ends

    def extract_tactical_trajectories(
        self,
        max_round_seconds: int = 45,
        step_seconds: float = 1.0,
        fields: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Extrait les trajectoires sous-échantillonnées pour les premières secondes
        de chaque round (phase tactique / setups / exécutions).

        :param max_round_seconds: Durée analysée par round (ex: 45s).
        :param step_seconds: Intervalle d'échantillonnage (ex: 1.0s = 1 point par seconde).
        :param fields: Attributs des joueurs à extraire.
        :return: DataFrame Pandas nettoyé.
        """
        if fields is None:
            fields = self.DEFAULT_FIELDS

        header = self.get_header_info()
        map_name = header.get("map_name", "unknown")
        rounds_df = self.get_round_intervals()

        # Échantillonnage : calcul des ticks exacts à extraire
        ticks_per_step = int(self.tickrate * step_seconds)
        steps_count = int(max_round_seconds / step_seconds)

        round_tick_map = []
        target_ticks = []

        for _, row in rounds_df.iterrows():
            r_idx = int(row["round_index"])
            start_tick = int(row["tick"])
            
            for step in range(steps_count + 1):
                t = start_tick + (step * ticks_per_step)
                target_ticks.append(t)
                round_tick_map.append({
                    "tick": t,
                    "round_index": r_idx,
                    "round_start_tick": start_tick,
                    "round_seconds": round(step * step_seconds, 2)
                })

        timing_df = pd.DataFrame(round_tick_map)

        # Extraction ciblée via demoparser2 (beaucoup plus rapide que parser toute la démo)
        raw_ticks_df = self.parser.parse_ticks(fields, ticks=target_ticks)

        if raw_ticks_df.empty:
            return pd.DataFrame()

        # Fusion avec les métadonnées temporelles du round
        merged_df = pd.merge(raw_ticks_df, timing_df, on="tick", how="inner")

        # Nettoyage Pandas
        # 1. Ne garder que les joueurs actifs (ignorer spectateurs / slots non assignés)
        clean_df = merged_df[merged_df["team_name"].isin(["TERRORIST", "CT"])].copy()

        # 2. Ajout de métadonnées
        clean_df["map_name"] = map_name
        clean_df["demo_name"] = self.demo_path.stem

        # 3. Tri propre
        clean_df = clean_df.sort_values(by=["round_index", "round_seconds", "team_name", "name"]).reset_index(drop=True)

        return clean_df

    def process_and_save(
        self,
        output_dir: str = "data/processed",
        max_round_seconds: int = 45,
        step_seconds: float = 1.0
    ) -> Path:
        """
        Extrait et sauvegarde les trajectoires au format CSV.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        file_dest = out_path / f"{self.demo_path.stem}_trajectories.csv"
        
        print(f"[*] Traitement de la démo : {self.demo_path.name}")
        df = self.extract_tactical_trajectories(
            max_round_seconds=max_round_seconds,
            step_seconds=step_seconds
        )
        
        df.to_csv(file_dest, index=False)
        print(f"[+] Succès : {len(df):,} lignes extraites -> {file_dest}")
        print(f"[i] Rounds traités : {df['round_index'].nunique()} | Map : {df['map_name'].iloc[0] if not df.empty else 'N/A'}")
        
        return file_dest


if __name__ == "__main__":
    demo_file = "data/raw_demos/100-thieves-vs-heroic-m1-dust2.dem"
    extractor = CS2DemoExtractor(demo_file)
    output_parquet = extractor.process_and_save()