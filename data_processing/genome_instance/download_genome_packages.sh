#!/bin/bash
# Optimized Parallel Download for Bird Genome Packages from PRJNA545868

# Create a clean directory
mkdir -p bird10k_genome_packages
cd bird10k_genome_packages

# Fetch metadata for 265 filtered genome assemblies
echo "Fetching genome metadata..."
datasets summary genome accession PRJNA545868 --annotated --reference > genome_filtered.json

# Extract genome accessions
jq -r '.reports[].accession' genome_filtered.json > genome_accessions_filtered.txt

# Dynamically set parallel jobs based on RAM (Recommended: 20)
TOTAL_RAM=$(free -g | awk '/^Mem:/ {print $2}')
AVAILABLE_RAM=$(free -g | awk '/^Mem:/ {print $7}')

if [[ $AVAILABLE_RAM -gt 40 ]]; then
    JOBS=25  # Maximum speed if RAM is high
elif [[ $AVAILABLE_RAM -gt 30 ]]; then
    JOBS=20  # Balanced speed
else
    JOBS=15  # Lower speed to avoid memory issues
fi

echo "Using $JOBS parallel jobs for download."

# Define parallel download function
download_genome_package() {
    acc=$1
    echo "Downloading Genome Package: $acc"
    datasets download genome accession "$acc" \
        --include genome,genome_annotation,protein,rna,cds,gff,gbff,data-report \
        --filename "${acc}.zip"
    
    # Extract and organize files
    unzip -o "${acc}.zip" -d "$acc"
    rm "${acc}.zip"
}

# Run parallel downloads
echo "Starting parallel downloads for 265 genome packages..."
cat genome_accessions_filtered.txt | parallel -j $JOBS download_genome_package {}

echo "All genome packages downloaded successfully!"


