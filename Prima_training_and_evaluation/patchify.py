import torch
from positional_encodings.torch_encodings import PositionalEncoding3D, PositionalEncoding1D
import time
import random


class MedicalImagePatchifier(torch.nn.Module):
    """
    A PyTorch module that processes 3D medical imaging data by adding positional encodings to visual token embeddings.
    
    This class takes visual token embeddings and their corresponding 3D coordinates, then:
    1. Adds 3D positional encodings to each token
    2. Adds orientation information based on the input's shape
    3. Concatenates all features together
    
    Attributes:
        out_dim (int): Output dimension (input_dim + positional_encoding_dim)
        d (int): Dimension of 3D positional encoding
        p_enc (torch.Tensor): Pre-computed positional encodings for a 100x100x100 grid
    """

    def __init__(self, in_dim=1024, d=30):
        """
        Initialize the patchifier.
        
        Args:
            in_dim (int): Input dimension of the visual token embeddings
            d (int): Dimension of 3D positional encoding (must be divisible by 3)
        """
        super().__init__()
        self.out_dim = in_dim + d
        assert d % 3 == 0, "Positional encoding dimension must be divisible by 3"

        # Initialize 3D positional encoding for a 100x100x100 grid
        p_enc = PositionalEncoding3D(d)
        self.p_enc = p_enc(torch.zeros(1, 100, 100, 100,
                                       d))[0].view(1000000, d)
        self.p_enc.requires_grad = False
        self.d = d

    def forward(self, xs, coords):
        """
        Process the input tokens and their coordinates.
        
        Args:
            xs (list[torch.Tensor]): List of visual token embeddings
            coords (list[torch.Tensor]): List of corresponding 3D coordinates for each token
            
        Returns:
            list[torch.Tensor]: List of processed tokens with positional encodings and orientation information
        """
        # Handle legacy code case
        if len(self.p_enc) == 100:
            self.p_enc = self.p_enc.view(1000000, self.d)
            self.p_enc.requires_grad = False

        processed_tokens = []
        for i, x in enumerate(xs):
            # Get input shape and determine orientation
            shapes = x.size()
            orientation = torch.zeros(3)  # Orientation embedding vector

            # Determine the orientation and division factors based on input shape
            if shapes[2] == 2:  # First dimension is small
                orientation[0] = 1
                div1, div2, div3 = 4, 32, 32
            elif shapes[3] == 2:  # Second dimension is small
                orientation[1] = 1
                div1, div2, div3 = 32, 4, 32
                x = x.transpose(2, 3)
            else:  # Third dimension is small
                assert shapes[4] == 2
                orientation[2] = 1
                div1, div2, div3 = 32, 32, 4
                x = x.transpose(2, 4)

            # Create division tensor for coordinate processing
            div_tensor = torch.tensor([div1, div2, div3],
                                      dtype=torch.long).unsqueeze(0)

            # Process coordinates to get positional encoding indices
            div_coords = coords[i] // div_tensor
            pos_enc_indices = div_coords[:,
                                         0] * 10000 + div_coords[:,
                                                                 1] * 100 + div_coords[:,
                                                                                       2]

            # Get positional encodings and orientation information
            pos_encodings = self.p_enc[torch.LongTensor(pos_enc_indices)]
            orientation_info = orientation.repeat(shapes[0], 1)

            # Concatenate all features
            processed_token = torch.cat(
                [
                    x.flatten(start_dim=1),  # Flattened visual tokens
                    pos_encodings.cpu(),  # Positional encodings
                    orientation_info  # Orientation information
                ],
                dim=1)

            processed_tokens.append(processed_token)

        return processed_tokens
