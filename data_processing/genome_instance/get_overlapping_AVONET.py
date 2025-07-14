import pandas as pd

# Load the Excel file
file_path = "./AVONET Supplementary dataset 1.xlsx"
xls = pd.ExcelFile(file_path)

# Load the AVONET1_BirdLife sheet
df_avonet1 = xls.parse("AVONET1_BirdLife")

# Load the list of 201 species
overlapping_species_path = "overlapping_species.csv"
overlapping_species_df = pd.read_csv(overlapping_species_path)

# Standardize species names for matching
overlapping_species_df["Scientific_Name"] = overlapping_species_df["Scientific_Name"].str.strip()
df_avonet1["Species1"] = df_avonet1["Species1"].str.strip()

# Perform the merge to find overlapping species and their phenotypes
matched_species_df = df_avonet1[df_avonet1["Species1"].isin(overlapping_species_df["Scientific_Name"])]
unmatched_species_df = overlapping_species_df[~overlapping_species_df["Scientific_Name"].isin(df_avonet1["Species1"])]
print(unmatched_species_df)
print(unmatched_species_df.shape)

# Save the result to CSV
output_path = "bird_AVONET_pointNet.csv"
matched_species_df.to_csv(output_path, index=False)
