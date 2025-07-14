import torch
import os
import time
import pickle
import concurrent.futures
import multiprocessing
from pathlib import Path

# -------------------- CONFIGURATION -------------------- #

INPUT_DIR = Path("output_embeddings")  # Directory with per-protein embeddings
OUTPUT_DIR = Path("species_pickles")  # Where individual species `.pkl` files are stored
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # Ensure the output directory exists
NUM_WORKERS = min(8, os.cpu_count())  # Use up to 8 CPUs per task

# -------------------- FUNCTION TO PROCESS EACH SPECIES -------------------- #

def load_protein_embedding(protein_file):
    """Loads a single protein embedding and returns (gene_name, embedding)."""
    data = torch.load(protein_file, map_location="cpu")  # Load .pt file into CPU
    mean_embedding = data["mean_representations"][6]  # Modify if using another layer
    gene_name = protein_file.stem  # Extract protein/gene name
    return gene_name, mean_embedding

def process_species(species_dir):
    """Processes a single species, loads all embeddings, and saves as a pickle file."""
    if not species_dir.is_dir():
        return None

    species_name = species_dir.name
    protein_files = list(species_dir.glob("*.pt"))

    if not protein_files:
        return None  # Skip if no protein files

    print(f"Processing {species_name} ({len(protein_files)} proteins)...")

    # Load all embeddings in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = list(executor.map(load_protein_embedding, protein_files))

    gene_names, embeddings = zip(*results)  # Split into separate lists

    # Stack into Nx320 tensor
    stacked_embeddings = torch.stack(embeddings)  # Shape: (N, 320)

    # Save per-species dictionary as a pickle file
    species_data = {"embeddings": stacked_embeddings, "gene_names": list(gene_names)}
    species_file = OUTPUT_DIR / f"{species_name}.pkl"

    with open(species_file, "wb") as f:
        pickle.dump(species_data, f)

    print(f"Saved {species_name} to {species_file}")

    return species_name

# -------------------- RUN PARALLEL SPECIES PROCESSING -------------------- #

if __name__ == "__main__":
    start_time = time.time()

    species_dirs = [d for d in INPUT_DIR.iterdir() if d.is_dir()]

    with multiprocessing.Pool(NUM_WORKERS) as pool:
        pool.map(process_species, species_dirs)

    print(f"--- Step 1 Completed: Saved individual species pickles in {OUTPUT_DIR} ---")
    print(f"--- Total runtime: {time.time() - start_time:.2f} seconds ---")
