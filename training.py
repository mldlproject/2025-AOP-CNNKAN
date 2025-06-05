import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, matthews_corrcoef, f1_score
from process_data import process_dataset
from modules.model import CombinedModel
import matplotlib.pyplot as plt
from tqdm import tqdm

class ProteinDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx].unsqueeze(0), self.labels[idx]  # Add channel dimension for CNN

def calculate_metrics(y_true, y_pred_proba):
    y_pred = (y_pred_proba > 0.5).astype(int)
    auc = roc_auc_score(y_true, y_pred_proba)
    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return {
        'auc': auc,
        'acc': acc,
        'mcc': mcc,
        'f1': f1
    }

def train_model(model, train_loader, val_loader, criterion, optimizer, device, 
                num_epochs=100, patience=10):
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_steps = 0
        
        for batch_x, batch_y in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_steps += 1
        
        avg_train_loss = train_loss / train_steps
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_steps = 0
        all_val_preds = []
        all_val_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                val_steps += 1
                
                probs = torch.softmax(outputs, dim=1)[:, 1]
                all_val_preds.extend(probs.cpu().numpy())
                all_val_labels.extend(batch_y.cpu().numpy())
        
        avg_val_loss = val_loss / val_steps
        val_losses.append(avg_val_loss)
        
        # Calculate validation metrics
        val_metrics = calculate_metrics(np.array(all_val_labels), np.array(all_val_preds))
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')
        print(f'Validation Metrics:')
        print(f'AUC: {val_metrics["auc"]:.4f}, ACC: {val_metrics["acc"]:.4f}')
        print(f'MCC: {val_metrics["mcc"]:.4f}, F1: {val_metrics["f1"]:.4f}')
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
    
    # # Plot training and validation losses
    # plt.figure(figsize=(10, 5))
    # plt.plot(train_losses, label='Training Loss')
    # plt.plot(val_losses, label='Validation Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Training and Validation Losses')
    # plt.legend()
    # plt.savefig('training_curves.png')
    # plt.close()
    
    return best_model_state

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    metrics = calculate_metrics(np.array(all_labels), np.array(all_preds))
    return metrics

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Check for CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load and process data
    print('Loading and processing data...')
    X, y = process_dataset('dataset/Anti-protein.txt', 'dataset/Non-Anti-protein.txt')
    
    # Create dataset
    dataset = ProteinDataset(X, y)
    
    # Split dataset
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Initialize model
    model = CombinedModel(num_classes=2).to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train model
    print('Starting training...')
    best_model_state = train_model(
        model, train_loader, val_loader, criterion, optimizer, device,
        num_epochs=50, patience=10
    )
    
    # Load best model and evaluate on test set
    model.load_state_dict(best_model_state)
    test_metrics = evaluate_model(model, test_loader, device)
    
    print('\nTest Set Results:')
    print(f'AUC: {test_metrics["auc"]:.4f}')
    print(f'Accuracy: {test_metrics["acc"]:.4f}')
    print(f'MCC: {test_metrics["mcc"]:.4f}')
    print(f'F1 Score: {test_metrics["f1"]:.4f}')
    
    # Save the best model
    torch.save(best_model_state, 'best_model.pth')

if __name__ == '__main__':
    main()
