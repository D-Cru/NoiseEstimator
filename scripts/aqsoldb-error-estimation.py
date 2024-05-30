# run this code with the raw data available at
# https://doi.org/10.24433/CO.1992938.v1

import pandas as pd
import merge
import descriptors
import numpy as np
import math
import matplotlib.pyplot as plt

data_a = pd.read_csv("../data/dataset-A.csv", header=0)
data_b = pd.read_csv("../data/dataset-B.csv", header=0)
data_c = pd.read_csv("../data/dataset-C.csv", header=0)
data_d = pd.read_csv("../data/dataset-D.csv", header=0)
data_e = pd.read_csv("../data/dataset-E.csv", header=0)
data_f = pd.read_csv("../data/dataset-F.csv", header=0)
data_g = pd.read_csv("../data/dataset-G.csv", header=0)
data_h = pd.read_csv("../data/dataset-H.csv", header=0)
data_i = pd.read_csv("../data/dataset-I.csv", header=0)


# first merge data directly
data_merged = pd.concat(
    [data_a, data_b, data_c, data_d, data_e, data_f, data_g, data_h, data_i]
)
data_merged = data_merged.reset_index(drop=True)

smiles_list = data_merged["SMILES"].tolist()
solubility_list = data_merged["Solubility"].tolist()
id_list = data_merged["ID"].tolist()
inchi_list = data_merged["InChI"].tolist()
name_list = data_merged["Name"].tolist()
prediction_list = data_merged["Prediction"].tolist()

# define variables and assign default values
dif_val_list = []
dif_sol_list = []

same_value_counter = 0  # same molecules with same values
different_value_counter_2times = (
    0  # same molecules with different values (2 occurences)
)
different_value_counter_mutiple = (
    0  # same molecules with different values (more than 2 occurences)
)

ocurrence_count = [-999] * len(id_list)
SD = [-999] * len(id_list)
reliability_group = ["-"] * len(id_list)
selected_list = [0] * len(id_list)

deviations = []

SD_dev = 0
n = 0
mol_with_duplicates = 0

print(len(id_list), "id_list")

# Second step: Select the most reliable solubility value among the same molecules (change unselected SMILES into XXX)
for i in range(0, len(id_list)):
    same_molecule_List = []

    # collect same molecules with different solubility value
    if smiles_list[i] != "XXX" and selected_list[i] == 0:
        same_molecule_List.append(i)
        for j in range(i + 1, len(id_list)):
            if smiles_list[j] != "XXX" and inchi_list[i] == inchi_list[j]:
                same_molecule_List.append(j)
                SD_dev += (solubility_list[i] - solubility_list[j]) ** 2
                deviations.append(solubility_list[i] - solubility_list[j])
                n += 1
    if len(same_molecule_List) > 1:
        mol_with_duplicates += 1
SD_dev = math.sqrt(SD_dev / (2 * (n - 1)))
print(SD_dev)
print(n)
print(len(same_molecule_List))
print(same_molecule_List)
print(len(deviations))
print(f"Based on {mol_with_duplicates} molecules with duplicates")

plt.hist(deviations, bins=50)
plt.savefig("deviations.png")
