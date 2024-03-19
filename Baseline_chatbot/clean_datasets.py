# %%
import os
import torch
import pickle
import torch.utils
import pandas as pd
from tqdm import tqdm

custom_header = ["user0", "user1", "movie", "lines"]
custom_header_lines = ["line_id", "user_id", "movie_id", "user_name", "line"]

movies_data = pd.read_csv(
    os.path.join("movie_dataset", "movie_conversations.tsv"),
    sep="\t",
    header=None,
    names=custom_header,
)  # reads the tsv movie conversations file

lines_data = pd.read_csv(
    os.path.join("movie_dataset", "movie_lines.tsv"),
    sep="\t",
    header=None,
    names=custom_header_lines,
    on_bad_lines="skip",
)
# Assuming you've already read the CSVs into movies_data and lines_data

# Drop rows with NaN values from lines_data
lines_data.dropna(inplace=True)
# %%

# Convert "lines" column in movies_data to a list of strings
movies_data["lines"] = movies_data["lines"].apply(
    lambda x: x.strip("[]").replace("'", "").split()
)
# %%

# Filter out dialogues where any line ID is not found in lines_data


# Preparing a set for faster membership testing
line_id_set = set(lines_data["line_id"].values)

# Initialize an empty list to hold indices of dialogues to keep
indices_to_keep = []

# Loop over movies_data with tqdm to see the progress
for i, row in tqdm(
    movies_data.iterrows(), total=movies_data.shape[0], desc="Filtering dialogues"
):
    # Check if all line IDs in the dialogue exist in lines_data
    if all(line in line_id_set for line in row["lines"]):
        indices_to_keep.append(i)

# Filter movies_data to only include dialogues with all lines present in lines_data
filtered_movies_data = movies_data.loc[indices_to_keep]

# lines_data.to_csv(
#     os.path.join("Filtered Datasets", f"filtered_movie_lines.tsv"),
#     sep="\t",
#     index=False,
# )

# filtered_movies_data.to_csv(
#     os.path.join("Filtered Datasets", f"filtered_movie_conversations.tsv"),
#     sep="\t",
#     index=False,
# )
# %%
