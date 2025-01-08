import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import glob
import os
import json


class LSTMGenerator(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.2,
    ):
        """
        Initializes the LSTM Generator model.

        Parameters:
        - vocab_size (int): Size of the vocabulary (number of unique tokens).
        - embed_size (int): Size of the embedding vector for each token.
        - hidden_size (int): Number of hidden units in the LSTM.
        - num_layers (int): Number of layers in the LSTM.
        - dropout (float): Dropout rate applied to the LSTM layers.
        """
        super(LSTMGenerator, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(
            embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor, prev_state: tuple) -> tuple:
        """
        Forward pass for generating predictions.

        Parameters:
        - x (torch.Tensor): Input tensor of word indices.
        - prev_state (tuple): The previous hidden and cell states of the LSTM.

        Returns:
        - tuple: Logits from the fully connected layer and the updated states.
        """
        word_embed = self.word_embedding(x)  # (batch_size, seq_length, embed_size)

        output, state = self.lstm(
            word_embed, prev_state
        )  # (batch_size, seq_length, lstm_size)
        logits = self.fc(output)  # (batch_size, seq_length, vocab_size)
        return logits, state

    def init_state(self, batch_size: int) -> tuple:
        """
        Initializes the hidden and cell states of the LSTM.

        Parameters:
        - batch_size (int): The batch size for generating sequences.

        Returns:
        - tuple: Initialized hidden and cell states.
        """
        return (
            torch.zeros(
                self.num_layers, batch_size, self.hidden_size, device=self.device
            ),
            torch.zeros(
                self.num_layers, batch_size, self.hidden_size, device=self.device
            ),
        )


def load_sequences(file_paths: list, word_to_idx: dict, seq_length: int) -> tuple:
    """
    Loads sequences and corresponding genres from file paths and prepares input-output pairs.

    Parameters:
    - file_paths (list): List of file paths containing text sequences.
    - word_to_idx (dict): Mapping from words to their corresponding indices.
    - seq_length (int): Length of the input sequence.

    Returns:
    - tuple: Input tokens and output tokens.
    """
    input_tokens, output_tokens = [], []
    for txt in file_paths:
        with open(txt, "r") as f:
            word_sequence = f.read().split(" ")
        if not word_sequence:
            continue
        seq_in, seq_out = prepare_sequences(word_sequence, word_to_idx, seq_length)
        input_tokens.extend(seq_in)
        output_tokens.extend(seq_out)
    return input_tokens, output_tokens


def build_vocab(dirname: str) -> tuple:
    """
    Builds vocabulary from a directory containing text files.

    Parameters:
    - dirname (str): Path to the directory containing text files.

    Returns:
    - tuple: Two dictionaries: word-to-index and index-to-word mappings.
    """
    word_count = {}
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


def prepare_sequences(word_sequence: list, word_to_idx: dict, seq_length: int) -> tuple:
    """
    Converts a word sequence into input-output pairs using a sliding window approach.

    Parameters:
    - word_sequence (list): List of words in the sequence.
    - word_to_idx (dict): Mapping from words to their indices.
    - seq_length (int): Length of the input sequence.

    Returns:
    - tuple: Input tokens and output tokens.
    """
    token_sequence = [
        word_to_idx[word] for word in word_sequence if word in word_to_idx
    ]
    input_tokens, output_tokens = [], []

    for i in range(len(token_sequence) - seq_length):
        input_tokens.append(token_sequence[i : i + seq_length])
        output_tokens.append(token_sequence[i + seq_length])
    return input_tokens, output_tokens


def create_loaders(
    input_tokens: list, output_tokens: list, batch_size: int
) -> DataLoader:
    """
    Creates DataLoader for training and testing.

    Parameters:
    - input_tokens (list): List of input token sequences.
    - output_tokens (list): List of output token sequences.
    - batch_size (int): Batch size for the DataLoader.

    Returns:
    - DataLoader: DataLoader for the dataset.
    """
    inputs = torch.LongTensor(input_tokens)
    outputs = torch.LongTensor(output_tokens)

    dataset = TensorDataset(inputs, outputs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_model(
    model: LSTMGenerator,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int,
    models_path: str,
    lr: float = 0.001,
    log_interval: int = 1000,
) -> tuple:
    """
    Trains the LSTM model and saves the best-performing models.

    Parameters:
    - model (LSTMGenerator): The LSTM model to be trained.
    - train_loader (DataLoader): DataLoader for the training dataset.
    - test_loader (DataLoader): DataLoader for the testing dataset.
    - num_epochs (int): Number of training epochs.
    - models_path (str): Path to save the trained models and losses.
    - lr (float): Learning rate for the optimizer.
    - log_interval (int): Interval for logging training progress.

    Returns:
    - tuple: Training and testing losses.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(model.device)

    test_loss_min = np.Inf
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
            test_loss_min = min(test_losses)
            train_loss_min = min(train_losses)
            print(f"Resuming from epoch {start_epoch + 1}...")

        # Load last saved model
        model.load_state_dict(torch.load(last_model_path))
        print(f"Loaded model from {last_model_path}.")

    for epoch in range(start_epoch, num_epochs):
        total_train_loss = 0
        model.train()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()

            inputs, targets = inputs.to(model.device), targets.to(model.device)

            state_h, state_c = model.init_state(inputs.size(0))
            outputs, (state_h, state_c) = model(inputs, (state_h, state_c))
            outputs_last_step = outputs[:, -1, :]
            state_h, state_c = state_h.detach(), state_c.detach()

            loss = criterion(outputs_last_step, targets)
            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()

            if batch_idx % log_interval == 0:
                print(
                    f"Epoch {epoch + 1}/{num_epochs} \t[Batch {batch_idx}/{len(train_loader)}] \tTraining Loss: {loss.item():.4f}"
                )

        train_loss = total_train_loss / len(train_loader)
        train_losses.append(train_loss)

        total_test_loss = 0
        with torch.no_grad():
            model.eval()
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(model.device), targets.to(model.device)
                state_h, state_c = model.init_state(inputs.size(0))

                outputs, (state_h, state_c) = model(inputs, (state_h, state_c))
                outputs_last_step = outputs[:, -1, :]
                state_h, state_c = state_h.detach(), state_c.detach()

                test_loss = criterion(outputs_last_step, targets)
                total_test_loss += test_loss.item()

        test_loss = total_test_loss / len(test_loader)
        test_losses.append(test_loss)

        print(
            f"Epoch {epoch + 1}/{num_epochs} \tTraining Loss: {train_loss:.3f} \tTest Loss: {test_loss:.3f}"
        )

        # Save models if losses improve
        if test_loss <= test_loss_min:
            print(
                f"Test loss decreased ({test_loss_min:.3f} -> {test_loss:.3f}). Saving model..."
            )
            torch.save(model.state_dict(), test_model_path)
            test_loss_min = test_loss

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
