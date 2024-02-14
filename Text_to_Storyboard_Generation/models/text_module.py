import torch
import torch.nn as nn

class TextUnderstandingModule(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(TextUnderstandingModule, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, text):
        embedded = self.embedding(text)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = lstm_out[:, -1, :]
        out = self.fc(lstm_out)
        out = self.relu(out)
        return out
