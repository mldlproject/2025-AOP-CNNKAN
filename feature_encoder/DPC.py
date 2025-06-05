import numpy as np
from collections import Counter

def calculate_dpc(sequence):
    """Calculate Dipeptide Composition (DPC) features"""
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    dipeptides = [aa1 + aa2 for aa1 in amino_acids for aa2 in amino_acids]
    
    # Count dipeptides
    dpc_counts = Counter()
    for i in range(len(sequence) - 1):
        dipeptide = sequence[i:i+2]
        if all(aa in amino_acids for aa in dipeptide):
            dpc_counts[dipeptide] += 1
    
    # Calculate frequencies
    total = sum(dpc_counts.values()) if sum(dpc_counts.values()) > 0 else 1
    dpc_features = [dpc_counts[dipeptide]/total for dipeptide in dipeptides]
    
    return np.array(dpc_features)