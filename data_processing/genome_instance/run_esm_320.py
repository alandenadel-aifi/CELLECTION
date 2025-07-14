import os
import time
import json
import pandas as pd
import torch
import multiprocessing
from pathlib import Path
from esm import pretrained, FastaBatchedDataset, MSATransformer

# -------------------- CONFIGURATION -------------------- #

# Base directories
BASE_DIR = Path("bird10k_genome_packages")  # Modify this path
OUTPUT_BASE_DIR = Path("output_embeddings")  # Modify output directory

# Model selection
MODEL_NAME = "esm2_t6_8M_UR50D"  # Modify based on model choice

# Number of parallel workers
NUM_WORKERS = 8  # Adjust based on available CPUs/GPUs

# -------------------- FIND SPECIES AND FASTA FILES -------------------- #

def get_species_info():
    """Extracts genome ID and scientific names from JSONL files in GCA_* directories."""
    species_data = []

    for genome_dir in BASE_DIR.glob("GCA_*"):
        jsonl_file = genome_dir / "ncbi_dataset/data/assembly_data_report.jsonl"
        protein_fasta = genome_dir / f"ncbi_dataset/data/{genome_dir.name}/protein.faa"

        if jsonl_file.exists() and protein_fasta.exists():
            with open(jsonl_file, "r") as f:
                data = json.load(f)
                scientific_name = data.get("organism", {}).get("organismName", "Unknown")
                species_data.append((genome_dir.name, scientific_name, protein_fasta))
    
    return species_data

species_list = get_species_info()

# -------------------- ESM EMBEDDING FUNCTION -------------------- #

def run_esm_embedding(model_name, fasta_file, output_dir):
    """Runs the ESM model on a given FASTA file and saves embeddings."""
    model, alphabet = pretrained.load_model_and_alphabet(model_name)
    model.eval()

    if isinstance(model, MSATransformer):
        raise ValueError("MSA Transformer models are not supported.")

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset = FastaBatchedDataset.from_file(fasta_file)
    batches = dataset.get_batch_indices(4096, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(1022), batch_sampler=batches
    )

    print(f"Processing {fasta_file} with {len(dataset)} sequences")
    output_dir.mkdir(parents=True, exist_ok=True)

    repr_layers = [0, 5, 6]  # Modify layers if needed
    repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in repr_layers]

    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(f"Batch {batch_idx + 1} of {len(batches)} ({toks.size(0)} sequences)")

            toks = toks.to(device, non_blocking=True)
            out = model(toks, repr_layers=repr_layers, return_contacts=False)

            representations = {layer: t.to("cpu") for layer, t in out["representations"].items()}

            for i, label in enumerate(labels):
                lab = label.split(".1")[0]
                output_file = output_dir / f"{lab}.pt"
                output_file.parent.mkdir(parents=True, exist_ok=True)

                result = {"label": label}
                truncate_len = min(1022, len(strs[i]))

                result["mean_representations"] = {
                    layer: t[i, 1:truncate_len + 1].mean(0).clone()
                    for layer, t in representations.items()
                }

                torch.save(result, output_file)

# -------------------- PROCESS EACH SPECIES -------------------- #

def process_species(species_info):
    """Processes a single species by running ESM embeddings on its FASTA file."""
    genome_id, scientific_name, fasta_file = species_info
    output_dir = OUTPUT_BASE_DIR / scientific_name.replace(" ", "_")

    print(f"Processing {scientific_name} ({genome_id})...")
    run_esm_embedding(MODEL_NAME, fasta_file, output_dir)

# -------------------- RUN PARALLEL PROCESSING -------------------- #

if __name__ == "__main__":
    start_time = time.time()

    # Use multiprocessing for parallel execution
    with multiprocessing.Pool(NUM_WORKERS) as pool:
        pool.map(process_species, species_list)

    print(f"--- Total runtime: {time.time() - start_time:.2f} seconds ---")
    #--- Total runtime: 78061.39 seconds --- for 3 Mil sequences #