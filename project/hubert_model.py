import torch
import torch.nn as nn
from transformers import HubertModel

class HuBERTSpectrogramModel(nn.Module):
    def __init__(self, num_classes):
        super(HuBERTSpectrogramModel, self).__init__()
        # Pretrained HuBERT model
        self.hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        self.fc = nn.Linear(self.hubert.config.hidden_size, num_classes)

    def forward(self, inputs):
        """
        Forward pass through HuBERT.
        Args:
            inputs: Tensor (batch_size, feature_dim, time_steps)
        Returns:
            Class logits.
        """
        outputs = self.hubert(inputs).last_hidden_state  # HuBERT outputs
        return self.fc(outputs[:, 0, :])  # Use the [CLS] token
