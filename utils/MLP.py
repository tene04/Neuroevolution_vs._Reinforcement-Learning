import torch
import torch.nn as nn
import numpy as np

class MLP(nn.Module):

    def __init__(self, input_size, hidden_sizes, output_size):
        """
        Initialize the MLP architecture

        Args:
            input_size (int): Number of input neurons
            hidden_sizes ([int]): Sizes of hidden layers
            output_size (int): Number of output neurons
        """
        super().__init__()
        layer_sizes = [input_size] + hidden_sizes + [output_size]

        layers = []
        for i in range(len(layer_sizes) - 2):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))

        self.net = nn.Sequential(*layers)

    def forward(self, x, model='DQL'):
        """
        Forward pass through the network for neuroevolution

        Args:
            x (np.ndarray): Input vector
            model (stre): Model type, should be 'neuroevolution' or 'DQL' for this method
        """
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        if model == 'neuroevolution':
            return self.net(x).detach().numpy()
        elif model == 'DQL':
            return self.net(x)
        else:
            raise ValueError("Model type must be 'neuroevolution' or 'DQL'.")
    
    def get_weights(self):
        """
        Flatten and return all network parameters
        """
        flat_weights = []
        for param in self.net.parameters():
            flat_weights.append(param.detach().cpu().numpy().flatten())
        return np.concatenate(flat_weights)

    def set_weights(self, weights):
        """
        Set all network parameters from a flat weight array

        Args:
            weights (np.ndarray): Flat array containing weights and biases
        """
        idx = 0
        for param in self.net.parameters():
            shape = param.data.shape
            size = np.prod(shape)
            new_vals = weights[idx:idx + size].reshape(shape)
            param.data = torch.tensor(new_vals, dtype=torch.float32)
            idx += size
