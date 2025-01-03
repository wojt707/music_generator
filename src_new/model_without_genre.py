import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import glob
import random
import os
import json


class LSTMGenerator(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        hidden_size: int,
        num_layers: int,
        dropout=0.2,
    ):
        super(LSTMGenerator, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(
            embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        word_embed = self.word_embedding(x)  # (batch_size, seq_length, embed_size)

        output, _ = self.lstm(word_embed)
        output = self.fc(output)
        return output[:, -1, :]


def load_sequences(file_paths, word_to_idx, seq_length):
    """
    Loads sequences and corresponding genres from file paths and prepares input-output pairs.
    """
    input_tokens, output_tokens = [], []
    for txt in file_paths:
        with open(txt, "r") as f:
            word_sequence = f.read().split(" ")
        seq_in, seq_out = prepare_sequences(word_sequence, word_to_idx, seq_length)
        input_tokens.extend(seq_in)
        output_tokens.extend(seq_out)
    return input_tokens, output_tokens


def build_vocab(dirname):
    word_count = dict()
    for txt in glob.glob(f"{dirname}/**/*.txt", recursive=True):
        word_sequence = []
        with open(txt, "r") as f:
            word_sequence = f.read().split(" ")
        for word in word_sequence:
            if word not in word_count:
                word_count[word] = 1
            else:
                word_count[word] += 1

    sorted_vocab = sorted(
        word_count.keys(), key=lambda word: word_count[word], reverse=True
    )
    word_to_idx = {word: idx for idx, word in enumerate(sorted_vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    return word_to_idx, idx_to_word


def prepare_sequences(word_sequence, word_to_idx, seq_length):
    """
    Converts a word sequence into input-output pairs using a sliding window approach.
    """
    token_sequence = [
        word_to_idx[word] for word in word_sequence if word in word_to_idx
    ]
    input_tokens, output_tokens = [], []

    for i in range(len(token_sequence) - seq_length):
        input_tokens.append(token_sequence[i : i + seq_length])
        output_tokens.append(token_sequence[i + seq_length])
    return input_tokens, output_tokens


def create_loaders(input_tokens, output_tokens, batch_size):

    inputs = torch.LongTensor(input_tokens)
    outputs = torch.LongTensor(output_tokens)

    dataset = TensorDataset(inputs, outputs)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return data_loader


def train_model(
    model,
    train_loader,
    test_loader,
    num_epochs,
    models_path,
    lr=0.001,
    log_interval=100,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device)

    valid_loss_min = np.Inf
    train_loss_min = np.Inf
    start_epoch = 0
    train_losses, test_losses = [], []

    # Check for existing saved state
    test_model_path = os.path.join(models_path, "test_min_model.pt")
    train_model_path = os.path.join(models_path, "train_min_model.pt")
    last_model_path = os.path.join(models_path, "last_model.pt")
    losses_path = os.path.join(models_path, "losses.json")

    if os.path.exists(losses_path) and os.path.exists(last_model_path):
        with open(losses_path, "r") as f:
            saved_data = json.load(f)
            train_losses = saved_data.get("train_losses", [])
            test_losses = saved_data.get("test_losses", [])
            start_epoch = len(train_losses)
            valid_loss_min = min(test_losses)
            train_loss_min = min(train_losses)
            print(f"Resuming from epoch {start_epoch + 1}...")

        # Load last saved model
        model.load_state_dict(torch.load(last_model_path))
        print(f"Loaded model from {last_model_path}.")

    for epoch in range(start_epoch, num_epochs):
        total_train_loss = 0
        model.train()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_train_loss += loss.item()

            # Backward pass
            loss.backward()
            optimizer.step()

            if batch_idx % log_interval == 0:
                print(
                    f"Epoch {epoch + 1}/{num_epochs} \t[Batch {batch_idx}/{len(train_loader)}] \tTraining Loss: {loss.item():.4f}"
                )

        total_test_loss = 0
        with torch.no_grad():
            model.eval()
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                test_loss = criterion(outputs, targets)
                total_test_loss += test_loss.item()

        train_loss = total_train_loss / len(train_loader)
        test_loss = total_test_loss / len(test_loader)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        print(
            f"Epoch {epoch + 1}/{num_epochs} \tTraining Loss: {train_loss:.3f} \tTest Loss: {test_loss:.3f}"
        )

        # Save models if losses improve
        if test_loss <= valid_loss_min:
            print(
                f"Validation loss decreased ({valid_loss_min:.3f} -> {test_loss:.3f}). Saving model..."
            )
            torch.save(model.state_dict(), test_model_path)
            valid_loss_min = test_loss

        if train_loss <= train_loss_min:
            print(
                f"Training loss decreased ({train_loss_min:.3f} -> {train_loss:.3f}). Saving model..."
            )
            torch.save(model.state_dict(), train_model_path)
            train_loss_min = train_loss
        torch.save(model.state_dict(), last_model_path)

        # Save losses
        with open(losses_path, "w") as f:
            json.dump({"train_losses": train_losses, "test_losses": test_losses}, f)

    return train_losses, test_losses
