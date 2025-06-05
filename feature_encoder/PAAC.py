import math

# Amino acid order
AA = 'ARNDCQEGHILKMFPSTWYV'
AADict = {aa: i for i, aa in enumerate(AA)}

# Properties from user-provided PAAC.txt table
raw_properties = {
    "Hydrophobicity": [0.62, -2.53, -0.78, -0.90, 0.29, -0.85, -0.74, 0.48, -0.40, 1.38,
                       1.06, -1.5, 0.64, 1.19, 0.12, -0.18, -0.05, 0.81, 0.26, 1.08],
    "Hydrophilicity": [-0.5, 3, 0.2, 3, -1, 0.2, 3, 0, -0.5, -1.8,
                       -1.8, 3, -1.3, -2.5, 0, 0.3, -0.4, -3.4, -2.3, -1.5],
    "SideChainMass":  [15, 101, 58, 59, 47, 72, 73, 1, 82, 57,
                       57, 73, 75, 91, 42, 31, 45, 130, 107, 43],
}

# Normalize each property vector to zero mean, unit variance
def normalize_properties(properties):
    normalized = []
    for prop in properties.values():
        mean = sum(prop) / len(prop)
        std = math.sqrt(sum((x - mean) ** 2 for x in prop) / len(prop))
        normalized.append([(x - mean) / std for x in prop])
    return normalized

normalized_properties = normalize_properties(raw_properties)

# Calculate Rvalue between two amino acids based on 3 normalized properties
def Rvalue(aa1, aa2):
    if aa1 not in AADict or aa2 not in AADict:
        return 0
    idx1, idx2 = AADict[aa1], AADict[aa2]
    return sum((prop[idx1] - prop[idx2]) ** 2 for prop in normalized_properties) / len(normalized_properties)

# Main PAAC feature calculator
def compute_paac(sequence, lambda_value=15, w=0.05):
    sequence = sequence.upper()
    sequence = ''.join([aa for aa in sequence if aa in AADict])
    L = len(sequence)
    if L < lambda_value + 1:
        raise ValueError(f"Sequence too short. Must be at least {lambda_value + 1} residues.")

    # Compute θ (theta) values for lambda = 1..λ
    theta = []
    for lam in range(1, lambda_value + 1):
        r_sum = sum(Rvalue(sequence[i], sequence[i + lam]) for i in range(L - lam))
        theta.append(r_sum / (L - lam))

    # AAC: Amino acid composition part (20 values)
    aac = [sequence.count(aa) / (1 + w * sum(theta)) for aa in AA]

    # Weighted sequence correlation part (15 values)
    theta_weighted = [(w * t) / (1 + w * sum(theta)) for t in theta]

    return aac + theta_weighted  # total: 20 + λ = 35 features

