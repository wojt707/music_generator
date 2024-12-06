import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import glob
import random
import os


class LSTMGenerator(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        genre_size: int,
        embed_size: int,
        hidden_size: int,
        num_layers: int,
        dropout=0.2,
    ):
        super(LSTMGenerator, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.genre_embedding = nn.Embedding(genre_size, embed_size)
        self.lstm = nn.LSTM(
            embed_size * 2, hidden_size, num_layers, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, genre):
        word_embed = self.word_embedding(x)  # (batch_size, seq_length, embed_size)
        genre_embed = self.genre_embedding(genre).unsqueeze(
            1
        )  # (batch_size, 1, embed_size)
        genre_embed = genre_embed.expand(
            -1, word_embed.size(1), -1
        )  # Match sequence length
        combined = torch.cat(
            (word_embed, genre_embed), dim=2
        )  # (batch_size, seq_length, embed_size*2)
        output, _ = self.lstm(combined)
        output = self.fc(output)
        return output


def split_dataset(txt_file_paths, split_ratio=0.9):
    """
    Splits file paths into training and testing sets.
    :param txt_file_paths: List of file paths.
    :param split_ratio: Ratio of data to use for training.
    :return: train_files, test_files
    """
    random.shuffle(txt_file_paths)
    split_index = int(len(txt_file_paths) * split_ratio)
    return txt_file_paths[:split_index], txt_file_paths[split_index:]


def load_sequences(file_paths, word_to_idx, genre_to_idx, seq_length):
    """
    Loads sequences and corresponding genres from file paths and prepares input-output pairs.
    """
    input_tokens, output_tokens, genres = [], [], []
    for txt in file_paths:
        with open(txt, "r") as f:
            word_sequence = f.read().split(" ")
        genre = genre_to_idx[get_genre_from_sequence_path(txt)]
        seq_in, seq_out = prepare_sequences(word_sequence, word_to_idx, seq_length)
        input_tokens.extend(seq_in)
        output_tokens.extend(seq_out)
        genres.extend([genre] * len(seq_in))
    return input_tokens, output_tokens, genres


def build_vocab(dirname):
    word_count = dict()
    for txt in glob.glob(f"{dirname}/**/*.txt", recursive=True):
        word_sequence = []
        with open(txt, "r") as f:
            word_sequence = f.read().split(" ")
        for word in word_sequence:
            if word not in word_count:
                word_count[word] = 0
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


def create_loaders(input_tokens, output_tokens, genres, batch_size):
    """
    Prepares batches of input-output pairs and genres for LSTM training.
    """

    inputs = torch.LongTensor(input_tokens)
    outputs = torch.LongTensor(output_tokens)
    genres = torch.LongTensor(genres)

    dataset = TensorDataset(inputs, genres, outputs)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return data_loader


def train_model(model, train_loader, test_loader, num_epochs, save_file, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device)

    valid_loss_min = np.Inf  # track change in validation loss

    train_losses, test_losses = [], []

    for epoch in range(num_epochs):
        total_train_loss = 0

        model.train()
        for inputs, genres, targets in train_loader:
            inputs, genres, targets = (
                inputs.to(device),
                genres.to(device),
                targets.to(device),
            )

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs, genres)
            outputs = outputs[:, -1, :]  # Get the last time step
            loss = criterion(outputs, targets)

            total_train_loss += loss.item()

            # Backward pass
            loss.backward()
            optimizer.step()
        else:
            total_test_loss = 0
            with torch.no_grad():
                model.eval()
                for inputs, genres, targets in test_loader:
                    inputs, genres, targets = (
                        inputs.to(device),
                        genres.to(device),
                        targets.to(device),
                    )

                    # Forward pass
                    outputs = model(inputs, genres)
                    outputs = outputs[:, -1, :]  # Get the last time step
                    test_loss = criterion(outputs, targets)

                    total_test_loss += test_loss.item()
            model.train()

            train_loss = total_train_loss / len(train_loader)
            test_loss = total_test_loss / len(test_loader)

            train_losses.append(train_loss)
            test_losses.append(test_loss)

        print(
            f"Epoch {epoch+1}/{num_epochs} \tTraining loss: {train_loss:.3f} \tTest loss: {test_loss:.3f}"
        )
        # save model if validation loss has decreased
        if test_loss <= valid_loss_min:
            print(
                f"Validation loss decreased ({valid_loss_min:.3f} --> {test_loss:.3f}).  Saving model ..."
            )
            torch.save(model.state_dict(), save_file)
            valid_loss_min = test_loss

    return train_losses, test_losses


def get_genre_from_sequence_path(path):
    """
    Extracts the genre from the file path.
    Assumes the genre is the first folder in the path structure.
    """
    return path.split(os.sep)[-5]


def generate_random_seed_sequence(word_to_idx, n=10):

    return random.sample(list(word_to_idx.keys()), n)


def generate_sequence(
    model, seed_sequence, word_to_idx, idx_to_word, seq_length, length=50
):
    """
    Generates a sequence of tokens using the trained model.
    """
    model.eval()
    generated_sequence = seed_sequence.copy()
    input_seq = [word_to_idx[word] for word in seed_sequence if word in word_to_idx]

    for _ in range(length):
        input_tensor = torch.tensor([input_seq[-seq_length:]], dtype=torch.long)

        with torch.no_grad():
            output = model(input_tensor)
            next_token = torch.argmax(output[:, -1, :], dim=-1).item()
            input_seq.append(next_token)
            generated_sequence.append(idx_to_word[next_token])

    return generated_sequence
