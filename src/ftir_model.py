
import torch
from torch import nn


class MaskedFTIRTransformer(nn.Module):

    def __init__(
        self,
        seq_len=257,
        d_model=128,
        nhead=4,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.1
    ):
        super().__init__()

        self.seq_len = seq_len
        self.d_model = d_model

        
        self.input_projection = nn.Linear(2, d_model)

        self.position_embedding = nn.Parameter(
            torch.zeros(1, seq_len, d_model)
        )

        nn.init.normal_(
            self.position_embedding,
            std=0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False
        )

        self.reconstruction_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )

    def encode(self, spectra, valid_mask, ssl_mask=None):
    
        if ssl_mask is None:
            ssl_mask = torch.zeros_like(valid_mask)

        features = torch.stack(
            [spectra, ssl_mask.float()],
            dim=-1
        )

        tokens = (
            self.input_projection(features)
            + self.position_embedding
        )

        encoded = self.encoder(
            tokens,
            src_key_padding_mask=~valid_mask
        )

        return encoded

    def forward(self, spectra, valid_mask, ssl_mask=None):
      
        encoded = self.encode(
            spectra,
            valid_mask,
            ssl_mask
        )

        reconstruction = self.reconstruction_head(
            encoded
        ).squeeze(-1)

        return reconstruction

    def get_embedding(self, spectra, valid_mask):
    
        encoded = self.encode(
            spectra,
            valid_mask
        )

        # Invalid tokens do not contribute to pooling.
        mask = valid_mask.unsqueeze(-1).float()

        summed = (encoded * mask).sum(dim=1)

        counts = mask.sum(dim=1).clamp(min=1.0)

        embedding = summed / counts

        return embedding
