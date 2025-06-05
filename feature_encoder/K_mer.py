import itertools
from collections import Counter
import numpy as np

def calculate_kmer(sequence, k=3):
    """Calculate k-mer composition features"""
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    kmers = [''.join(p) for p in itertools.product(amino_acids, repeat=k)]
    
    # Count k-mers
    kmer_counts = Counter()
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]
        if all(aa in amino_acids for aa in kmer):
            kmer_counts[kmer] += 1
    
    # Calculate frequencies
    total = sum(kmer_counts.values()) if sum(kmer_counts.values()) > 0 else 1
    kmer_features = [kmer_counts[kmer]/total for kmer in kmers]
    
    return np.array(kmer_features)