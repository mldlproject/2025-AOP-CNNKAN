import numpy as np
from feature_encoder.BPF import calculate_bpf
from feature_encoder.DPC import calculate_dpc
from feature_encoder.K_mer import calculate_kmer
from feature_encoder.PAAC import compute_paac

def read_sequences(file_path):
    """Read sequences from a file"""
    with open(file_path, 'r') as f:
        sequences = [line.strip() for line in f if line.strip() and len(line.strip()) > 16]
    return sequences

def generate_features(sequence):
    """Generate combined features for a single sequence"""
    # Skip sequences shorter than 16 amino acids
    if len(sequence) <= 16:
        return None
        
    # Generate individual features
    bpf_features = calculate_bpf(sequence)
    dpc_features = calculate_dpc(sequence)
    kmer_features = calculate_kmer(sequence, k=2)
    paac_features = compute_paac(sequence)
    
    # Concatenate all features
    combined_features = np.concatenate([
        bpf_features,
        dpc_features,
        kmer_features,
        paac_features
    ])
    
    return combined_features

def process_dataset(anti_file, non_anti_file):
    """Process both anti-protein and non-anti-protein datasets"""
    # Read sequences
    anti_sequences = read_sequences(anti_file)
    non_anti_sequences = read_sequences(non_anti_file)
    
    print(f"Number of anti-protein sequences (length > 16): {len(anti_sequences)}")
    print(f"Number of non-anti-protein sequences (length > 16): {len(non_anti_sequences)}")
    
    # Generate features
    anti_features = []
    for seq in anti_sequences:
        features = generate_features(seq)
        if features is not None:
            anti_features.append(features)
            
    non_anti_features = []
    for seq in non_anti_sequences:
        features = generate_features(seq)
        if features is not None:
            non_anti_features.append(features)
    
    # Create labels
    anti_labels = np.ones(len(anti_features))
    non_anti_labels = np.zeros(len(non_anti_features))
    
    # Combine features and labels
    X = np.vstack(anti_features + non_anti_features)
    y = np.concatenate([anti_labels, non_anti_labels])
    
    return X, y

if __name__ == "__main__":
    # Example usage
    anti_file = "dataset/Anti-protein.txt"
    non_anti_file = "dataset/Non-Anti-protein.txt"
    
    # Test with a single sequence first
    test_sequence = "MKKLLFAALLCLLQFSWTVPG"
    print("Testing feature generation with a single sequence:")
    print(f"Sequence: {test_sequence}")
    
    features = generate_features(test_sequence)
    print(f"\nFeature vector shape: {features.shape}")
    print(f"Feature vector (first 10 elements): {features[:10]}")
    
    # Process full dataset
    print("\nProcessing full dataset...")
    X, y = process_dataset(anti_file, non_anti_file)
    print(f"Full dataset shape: {X.shape}")
    print(f"Number of anti-proteins: {sum(y == 1)}")
    print(f"Number of non-anti-proteins: {sum(y == 0)}")
