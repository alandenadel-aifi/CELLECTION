import pickle
import os
from pathlib import Path

# -------------------- CONFIGURATION -------------------- #

INPUT_DIR = Path("species_pickles")  # Directory with per-species pickle files
OUTPUT_FILE = Path("final_aggregated_embeddings.pkl")  # Final merged pickle

# -------------------- MERGE ALL SPECIES PICKLES -------------------- #

def load_species_pickle(species_file):
    """Loads a single species pickle file."""
    with open(species_file, "rb") as f:
        return pickle.load(f)

if __name__ == "__main__":
    species_files = list(INPUT_DIR.glob("*.pkl"))

    aggregated_data = {}

    for species_file in species_files:
        species_name = species_file.stem
        print(f"Merging {species_name}...")
        aggregated_data[species_name] = load_species_pickle(species_file)

    # Save final merged dictionary
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(aggregated_data, f)

    print(f"--- Merged all species into {OUTPUT_FILE} ---")
