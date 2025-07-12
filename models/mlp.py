import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Simple MLP: Predict (1, C, H, W) from time series (T, C, H, W)
    Input: (batch_size, T, C, H, W) -> Output: (batch_size, 1, C, H, W)
    """
    
    def __init__(self, input_shape, hidden_size=256):
        """
        Initialize MLP model for time series prediction
        
        Args:
            input_shape: Shape of input tensor (T, C, H, W)
            hidden_size: Size of hidden layer
        """
        super(MLP, self).__init__()
        
        self.input_shape = input_shape  # (T, C, H, W)
        T, C, H, W = input_shape
        
        # Input size: flatten all dimensions T * C * H * W
        self.input_size = T * C * H * W
        # Output size: only C * H * W (predict 1 time step)
        self.output_size = C * H * W
        
        self.fc1 = nn.Linear(self.input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, self.output_size)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, T, C, H, W)
        
        Returns:
            Output tensor of shape (batch_size, 1, C, H, W)
        """
        # Store original shape info
        batch_size = x.shape[0]
        T, C, H, W = self.input_shape
        
        # Flatten all dimensions except batch: (batch_size, T*C*H*W)
        x = x.view(batch_size, -1)
        
        # Linear + ReLU + Linear
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        # Reshape to (batch_size, 1, C, H, W)
        x = x.view(batch_size, 1, C, H, W)
        
        return x


def test_mlp():
    """Test the MLP model for time series prediction"""
    print("Testing MLP for Time Series Prediction...")
    
    # Test parameters
    batch_size = 4
    T, C, H, W = 8, 10, 16, 16  # 8 time steps, 10 channels, 16x16 spatial
    input_shape = (T, C, H, W)
    
    # Create model and test input
    model = MLP(input_shape, hidden_size=512)
    x = torch.randn(batch_size, T, C, H, W)  # (batch_size, T, C, H, W)
    
    print(f"Input shape: {x.shape}")
    print(f"Input size (flattened): {T * C * H * W}")
    print(f"Output size (flattened): {C * H * W}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(x)
        print(f"Output shape: {output.shape}")
        expected_shape = (batch_size, 1, C, H, W)
        print(f"Expected shape: {expected_shape}")
        print(f"Shape match: {output.shape == expected_shape}")
        
        # Show the transformation
        print(f"\nTransformation details:")
        print(f"Input: {x.shape} -> Flattened: ({batch_size}, {T * C * H * W})")
        print(f"Output: ({batch_size}, {C * H * W}) -> Reshaped: {output.shape}")
        print(f"Time series prediction: {T} steps -> 1 step")


if __name__ == "__main__":
    test_mlp()



