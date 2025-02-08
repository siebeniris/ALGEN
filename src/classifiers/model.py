import torch.nn as nn
from src.defenses.ldp import PurMech, LapMech


class Classifier(nn.Module):
    def __init__(self, input_dim, num_labels, defense_method, epsilon, dropout_rate=0.2):
        super(Classifier, self).__init__()
        # dropout?
        self.input_dim = input_dim
        self.defense_method = defense_method
        self.epsilon = epsilon
        self.proj_dim = 16  # paper.

        # purmech, lapmech3
        # https://github.com/xiangyue9607/Sentence-LDP/blob/main/model.py
        if self.defense_method in ["PurMech", "LapMech"]:
            self.dropout = nn.Dropout(p=dropout_rate)
            if self.input_dim > 768:
                self.project_1 = nn.Linear(input_dim, 1024)
                self.project_2 = nn.Linear(1024, self.proj_dim)
                self.project_3 = nn.Linear(self.proj_dim, 1024)
                self.project_4 = nn.Linear(1024, input_dim)
            else:

                self.project_1 = nn.Linear(input_dim, self.proj_dim)
                self.project_2 = nn.Linear(self.proj_dim, input_dim)
                self.activation = nn.Tanh()
            self.classifier = nn.Linear(input_dim, num_labels)
        else:
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
        if self.defense_method in ["PurMech", "LapMech"]:
            x = self.dropout(x)

            if self.input_dim > 768:
                x = self.project_1(x)
                x = self.project_2(x)
                x = self.activation(x)
                if self.defense_method == "PurMech":
                    x = PurMech(x, self.epsilon)
                elif self.defense_method == "LapMech":
                    x = LapMech(x, self.epsilon)
                x = self.project_3(x)
                x = self.project_4(x)
            else:
                x = self.project_1(x)
                x = self.activation(x)
                if self.defense_method == "PurMech":
                    x = PurMech(x, self.epsilon)
                elif self.defense_method == "LapMech":
                    x = LapMech(x, self.epsilon)
                x = self.project_2(x)

            x = self.activation(x)
            x = self.classifier(x)
            return x
        else:
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
