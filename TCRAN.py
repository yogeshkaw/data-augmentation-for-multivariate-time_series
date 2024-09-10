import torch
import torch.nn as nn
import torch.nn.functional as F

class AFM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(AFM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
 
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return (x * y) + x

class IAM(nn.Module):
    def __init__(self, num_blocks, channels):
        super(IAM, self).__init__()
        self.num_blocks = num_blocks
        self.channels = channels

        # Adjust the convolution to handle num_blocks * channels as input
        self.conv = nn.Conv1d(num_blocks * channels, channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
 
    def forward(self, x):
        # x shape: (batch_size, num_blocks * channels, time_steps)
        batch_size, _, time_steps = x.size()

        # Reshape to (batch_size, num_blocks, channels * time_steps)
        x = x.view(batch_size, self.num_blocks, -1)

        # Compute correlation matrix
        correlation_matrix = torch.sigmoid(torch.matmul(x, x.transpose(1, 2)))

        # Multiply and add original matrix
        x = torch.matmul(correlation_matrix, x) + x

        # Reshape back to (batch_size, num_blocks * channels, time_steps)
        x = x.view(batch_size, -1, time_steps)

        # Apply convolution, batch normalization, and ReLU
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x



class TRCAB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TRCAB, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.afm = AFM(out_channels)
        self.bi_gru = nn.GRU(out_channels, out_channels, batch_first=True, bidirectional=True)

        # Linear layer to project back to out_channels after GRU
        self.proj_back = nn.Linear(out_channels * 2, out_channels)
 
    def forward(self, x):
        residual = x
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.afm(out)
        
        # Permute to match GRU input expectations (batch, seq_len, feature_dim)
        out = out.permute(0, 2, 1)
        out, _ = self.bi_gru(out)

        # Project back to the original number of channels
        out = self.proj_back(out)

        # Permute back to original shape (batch, feature_dim, seq_len)
        out = out.permute(0, 2, 1)
        
        out += residual
        return out



class ClassPrototypeAttention(nn.Module):
    def __init__(self, num_classes, embedding_dim):
        super(ClassPrototypeAttention, self).__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.prototypes = nn.Parameter(torch.randn(num_classes, embedding_dim))
 
    def forward(self, x):
        distances = torch.cdist(x.unsqueeze(1), self.prototypes.unsqueeze(0))
        return distances
    
class TCRAN(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_classes, num_blocks=3):
        super(TCRAN, self).__init__()

        self.conv = nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)

        self.trcabs = nn.ModuleList([TRCAB(hidden_dim, hidden_dim) for _ in range(num_blocks)])
        self.iam = IAM(num_blocks, hidden_dim)
        self.lstm = nn.LSTM(in_channels, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.class_prototype_attention = ClassPrototypeAttention(num_classes, hidden_dim)
 
    def forward(self, x):
        x = x.permute(0, 2, 1)
        batch_size, time_steps, channels = x.shape

        # CNN Block
        residual_lstm = x
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        
        residual_cnn = x
        input_iam = []

        for trcab in self.trcabs:
            x = trcab(x)
            input_iam.append(x)
        
        x = self.conv2(x)

        # Concatenate along the channel dimension for IAM input
        input_iam = torch.cat(input_iam, dim=1)  # Corrected to use torch.cat
        out_iam = self.iam(input_iam)

        x = residual_cnn + x + out_iam
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)  # Squeeze to remove time dimension
        
        # LSTM Block
        x2, _ = self.lstm(residual_lstm.permute(0, 2, 1))
        x2 = F.adaptive_avg_pool1d(x2.permute(0, 2, 1), 1).squeeze(-1)

        # Combine CNN and LSTM features
        x = x + x2

        # Final fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # Class Prototype Attention
        distances = self.class_prototype_attention(x)
        #probabilities = F.softmax(-distances, dim=-1)
        return distances
