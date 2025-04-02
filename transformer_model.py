import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, input_size, num_heads, num_layers, dim_feedforward, dropout):
        super(TransformerModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=num_layers
        )
        self.fc = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, src):
        src = self.transformer_encoder(src)
        output = self.fc(src[:, -1, :])
        output = self.sigmoid(output)
        return output