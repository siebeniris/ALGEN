import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, input_dim, num_labels, dropout_rate=0.2):
        super(Classifier, self).__init__()
        # dropout?
        self.input_dim = input_dim

        # for gpt models.
        if self.input_dim > 768:
            self.fc1 = nn.Linear(input_dim, 1024)  # Increased hidden units for larger embeddings
            self.fc2 = nn.Linear(1024, num_labels)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(p=dropout_rate)

            # Layer for smaller input (e.g., BERT embeddings)
        else:
            self.dropout = nn.Dropout(p=dropout_rate)
            self.fc = nn.Linear(input_dim, num_labels)


    def forward(self, x):
        if self.input_dim > 768:
            # print("Using the model specific for GPT embeddings")
            x = self.dropout(x)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x
        else:
            x = self.dropout(x)
            return self.fc(x)

