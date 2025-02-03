import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, input_dim, num_labels):
        super(Classifier, self).__init__()
        # dropout?
        self.fc = nn.Linear(input_dim, num_labels)

    def forward(self, x):
        return self.fc(x)
