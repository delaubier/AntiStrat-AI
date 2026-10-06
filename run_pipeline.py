"""
CS2 Tactical Scouting Engine - Master Pipeline Runner
Executes the full pipeline end-to-end:
Parsing -> Normalization -> Deep Learning Training -> Embeddings -> Tactical Clustering.
"""

from pathlib import Path
import json
import torch
import pandas as pd

from src.parser.demo_extractor import CS2DemoExtractor
from src.parser.map_normalizer import MapNormalizer
from src.models.dataset import CS2TacticalDataset, get_dataloader
from src.models.trainer import TacticalTrainer
from src.clustering.cluster_engine import TacticalClusterEngine
from src.clustering.vector_index import TacticalVectorIndex


def run_full_pipeline(
    demo_path: str = "data/raw_demos/100-thieves-vs-heroic-m1-dust2.dem",
    team_side: str = "TERRORIST",
    epochs: int = 30,
    n_clusters: int = 3
):
    print("=" * 70)
    print("[*] DEMARRAGE DU PIPELINE DE SCOUTING AUTOMATISE CS2")
    print("=" * 70)

    # 1. PARSING DE LA DEMO
    print("\n[PHASE 1] Ingestion et parsing binaire avec demoparser2...")
    extractor = CS2DemoExtractor(demo_path)
    csv_path = extractor.process_and_save()

    # 2. NORMALISATION ET PREPARATION PYTORCH
    print("\n[PHASE 2] Normalisation spatiale et creation du Dataset PyTorch...")
    dataset = CS2TacticalDataset(csv_path, team_side=team_side)
    print(f"[+] Dataset pret : {len(dataset)} manches ({team_side}) converties en tenseurs (46, 30)")
    loader = get_dataloader(csv_path, team_side=team_side, batch_size=4, shuffle=True)

    # 3. ENTRAINEMENT DU MODELE DE DEEP LEARNING (AUTOENCODEUR)
    print("\n[PHASE 3] Entrainement de l'Autoencodeur PyTorch...")
    trainer = TacticalTrainer(learning_rate=2e-3)
    trainer.train(loader, epochs=epochs, verbose=True)

    # 4. EXTRACTION DES EMBEDDINGS TACTIQUES
    print("\n[PHASE 4] Generation des Empreintes Tactiques (Embeddings)...")
    emb_df = trainer.extract_embeddings(dataset)
    print(f"[+] {len(emb_df)} embeddings de dimension 64 generes avec succes.")

    # 5. REGROUPEMENT DES STRATEGIES (CLUSTERING)
    print("\n[PHASE 5] Regroupement des strategies & Rapport de Scouting...")
    engine = TacticalClusterEngine(n_clusters=n_clusters)
    clustered_df = engine.fit_predict(emb_df)

    raw_trajectories = pd.read_csv(csv_path)
    raw_trajectories = MapNormalizer().normalize(raw_trajectories)
    report = engine.generate_scouting_report(clustered_df, raw_trajectories)

    print("\n" + "=" * 60)
    print(f"[RAPPORT FINAL DE SCOUTING] - {report['map_name'].upper()} ({report['team_side']})")
    print(f"Total des manches analysees : {report['total_rounds_analyzed']}")
    print("=" * 60)
    
    for c in report["clusters"]:
        print(f"\n> Strategie #{c['cluster_id'] + 1} : {c['tactical_label']}")
        print(f"  - Frequence : {c['frequency_percentage']}% ({c['rounds_count']} manches)")
        print(f"  - Manches : {c['rounds_list']}")

    report_path = Path("data/embeddings/tactical_scouting_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Rapport complet sauvegarde : {report_path}")

    # 6. MOTEUR DE RECHERCHE PAR SIMILARITE (DEMONSTRATION)
    print("\n[PHASE 6] Test de la recherche vectorielle de similarite...")
    index = TacticalVectorIndex(emb_df)
    first_round = emb_df["round_index"].iloc[0]
    similar = index.search_similar(query_round_index=first_round, top_k=3)
    print(f"[+] Top 3 des manches les plus similaires a la Manche #{first_round} :")
    for rank, res in enumerate(similar, 1):
        print(f"    #{rank} : Manche {res['round_index']} (Similarite tactique : {res['similarity_score']}%)")

    print("\n" + "=" * 70)
    print("[+] PIPELINE EXECUTE AVEC SUCCES - PRET POUR L'AUTOMATISATION AIRFLOW")
    print("=" * 70)


if __name__ == "__main__":
    run_full_pipeline()
