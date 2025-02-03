import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, input_dim, num_labels, dropout_rate=0.2):
        super(Classifier, self).__init__()
        # dropout?
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(input_dim, num_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.fc(x)
