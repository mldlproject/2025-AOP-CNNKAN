import torch
import torch.nn as nn
import torch.nn.functional as F
from .cnn import CNN
from .kan import KAN

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        
    def forward(self, q, k, v):
        # q, k, v shape: (batch_size, seq_len, d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)
        return output, attn

class CombinedModel(nn.Module):
    def __init__(self, num_classes=128, cnn_channels=1, kan_hidden=[1335, 512, 256], attention_dim=256):
        super(CombinedModel, self).__init__()
        
        # CNN branch
        self.cnn = CNN(input_channels=cnn_channels, num_classes=attention_dim)
        
        # KAN branch
        self.kan = KAN(
            layers_hidden=kan_hidden + [attention_dim],
            grid_size=5,
            spline_order=3
        )
        
        # Attention mechanism
        self.attention = ScaledDotProductAttention(d_k=attention_dim)
        
        # Projection layers for attention
        self.query_proj = nn.Linear(attention_dim, attention_dim)
        self.key_proj = nn.Linear(attention_dim, attention_dim)
        self.value_proj = nn.Linear(attention_dim, attention_dim)
        
        # Final classification layer
        self.fc = nn.Linear(attention_dim, num_classes)
        
    def forward(self, x):
        # Input shape: (batch_size, 1, 1335)
        
        # CNN branch
        cnn_out = self.cnn(x)  # (batch_size, attention_dim)
        
        # KAN branch - reshape input for KAN
        kan_input = x.squeeze(1)  # (batch_size, 1335)
        kan_out = self.kan(kan_input)  # (batch_size, attention_dim)
        
        # Prepare features for attention
        # Reshape to (batch_size, 1, attention_dim) for attention computation
        cnn_features = cnn_out.unsqueeze(1)
        kan_features = kan_out.unsqueeze(1)
        
        # Combine features for attention
        combined_features = torch.cat([cnn_features, kan_features], dim=1)  # (batch_size, 2, attention_dim)
        
        # Apply attention
        q = self.query_proj(combined_features)
        k = self.key_proj(combined_features)
        v = self.value_proj(combined_features)
        
        attended_features, _ = self.attention(q, k, v)
        
        # Take mean of attended features
        final_features = attended_features.mean(dim=1)  # (batch_size, attention_dim)
        
        # Final classification
        output = self.fc(final_features)
        
        return output
