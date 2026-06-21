"""Example: fetch Elo and FIFA rankings and save to CSV."""
from src.data import EloRatings, FifaRankings

elo = EloRatings()
elo.run()
import os

raw_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
os.makedirs(raw_dir, exist_ok=True)

elo.to_csv(os.path.join(raw_dir, "elo_ratings.csv"))
print("Saved elo_ratings.csv to data/raw/")

fifa = FifaRankings()
fifa.run()
fifa.to_csv(os.path.join(raw_dir, "fifa_rankings.csv"))
print("Saved fifa_rankings.csv to data/raw/")
