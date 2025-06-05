import numpy as np

def calculate_bpf(sequence, max_length=25):
    """Calculate Binary Profile Features (BPF)"""
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    
    # Initialize matrix
    bpf = np.zeros((max_length, len(amino_acids)))
    
    # Fill matrix
    for i, aa in enumerate(sequence[:max_length]):
        if aa in amino_acids:
            bpf[i, amino_acids.index(aa)] = 1
            
    return bpf.flatten()
