import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset
import transformer_model as TransformerModel
import feature_engineer as FeatureEngineer
import transformer_data_preparer as TransformerDataPreparer
from sklearn.model_selection import TimeSeriesSplit

class ModelTrainer:
    def __init__(self, df, sequence_length, epochs):
        self.df = df
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.best_metrics = {"train_loss": float('inf'), "train_accuracy": 0, "val_accuracy": 0}

    def create_dataloader(self, X, y, batch_size):
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
        dataloader = TorchDataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader

    def create_model(self, input_size, num_heads, num_layers, dim_feedforward, dropout):
        if input_size % num_heads != 0:
            raise ValueError(f"embed_dim (input_size) must be divisible by num_heads. Got embed_dim={input_size} and num_heads={num_heads}")
        
        model = TransformerModel(
            input_size=input_size,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        return model

    def train_model(self, model, train_dataloader, val_dataloader, learning_rate):
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        train_losses = []
        train_accuracies = []
        val_accuracies = []

        no_improvement_epochs = 0

        model.train()
        for epoch in range(self.epochs):
            train_loss = 0.0
            correct_train = 0
            total_train = 0
            for X_batch, y_batch in train_dataloader:
                optimizer.zero_grad()
                output = model(X_batch)
                loss = criterion(output.squeeze(), y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
                predicted = (output.squeeze() > 0.5).float()
                total_train += y_batch.size(0)
                correct_train += (predicted == y_batch).sum().item()

            train_loss /= len(train_dataloader)
            train_accuracy = correct_train / total_train
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)

            val_accuracy = self.evaluate_model(model, val_dataloader)
            val_accuracies.append(val_accuracy)

            print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}")

            # Update best metrics
            if val_accuracy > self.best_metrics["val_accuracy"]:
                self.best_metrics["train_loss"] = train_loss
                self.best_metrics["train_accuracy"] = train_accuracy
                self.best_metrics["val_accuracy"] = val_accuracy
                no_improvement_epochs = 0
            else:
                no_improvement_epochs += 1

            # Exit if no improvement in validation accuracy for 10 epochs
            if no_improvement_epochs > 50:
                print("Early stopping due to no improvement in validation accuracy for 10 epochs")
                break

        self.plot_metrics(train_losses, train_accuracies, val_accuracies)
        return train_accuracy, val_accuracy

    def evaluate_model(self, model, dataloader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                output = model(X_batch)
                predicted = (output.squeeze() > 0.5).float()
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        accuracy = correct / total
        return accuracy

    def plot_metrics(self, train_losses, train_accuracies, val_accuracies):
        epochs = range(1, len(train_losses) + 1)
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 3, 1)
        plt.plot(epochs, train_losses, label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(epochs, train_accuracies, label='Training Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy')
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.plot(epochs, val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Validation Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def objective(self, trial):
        # Тюнинг параметров фичей
        sma_length = trial.suggest_int('sma_length', 10, 200, step=10)
        ema_length = trial.suggest_int('ema_length', 10, 200, step=10)
        rsi_length = trial.suggest_int('rsi_length', 10, 50, step=5)
        atr_length = trial.suggest_int('atr_length', 10, 50, step=5)
        bb_length = trial.suggest_int('bb_length', 10, 50, step=5)
        stoch_length = trial.suggest_int('stoch_length', 10, 50, step=5)
        momentum_length = trial.suggest_int('momentum_length', 10, 50, step=5)
        adx_length = trial.suggest_int('adx_length', 10, 50, step=2)
        cci_length = trial.suggest_int('cci_length', 10, 50, step=2)
        volatility_length = trial.suggest_int('volatility_length', 10, 50, step=5)
        roc_length = trial.suggest_int('roc_length', 10, 50, step=5)
        williams_r_length = trial.suggest_int('williams_r_length', 10, 50, step=5)
        macd_slow = trial.suggest_int('macd_slow', 10, 50, step=2)
        macd_fast = trial.suggest_int('macd_fast', 10, 50, step=2)
        macd_signal = trial.suggest_int('macd_signal', 5, 50, step=2)
        lag_value = trial.suggest_int('lag_value', 5, 50, step=5)
        price_spike_threshold = trial.suggest_float('price_spike_threshold', 0.01, 0.05)

        feature_engineer = FeatureEngineer(self.df.copy())
        feature_engineer.advanced_features(
            sma_lengths=[sma_length],
            ema_lengths=[ema_length],
            rsi_lengths=[rsi_length],
            atr_lengths=[atr_length],
            bb_lengths=[bb_length],
            macd_combinations={'slow_values': [macd_slow], 'fast_values': [macd_fast], 'signal_values': [macd_signal]},
            stoch_lengths=[stoch_length],
            momentum_lengths=[momentum_length],
            adx_lengths=[adx_length],
            cci_lengths=[cci_length],
            volatility_lengths=[volatility_length],
            roc_lengths=[roc_length],
            williams_r_lengths=[williams_r_length],
            lag_value=lag_value,
            price_spike_threshold=price_spike_threshold
        )

        preparer = TransformerDataPreparer(sequence_length=self.sequence_length)
        X, y, scaler, dates, prices = preparer.prepare_data(feature_engineer.df)


        tscv = TimeSeriesSplit(n_splits = 5)
        accuracies = []

        for train_index, val_index in tscv.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            # Тюнинг гиперпараметров модели
            batch_size = trial.suggest_categorical('batch_size', [ 128, 256, 512 ])
            num_heads = trial.suggest_int('num_heads', 2, 12)
            num_layers = trial.suggest_int('num_layers', 2, 12)
            dim_feedforward = trial.suggest_int('dim_feedforward', 128, 256, 512)
            dropout = trial.suggest_float('dropout', 0.1, 0.5)
            learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)

            input_size = X.shape[2]
            if input_size % num_heads != 0:
                num_heads = 1  # Adjust num_heads to 1 if input_size is not divisible

            train_dataloader = self.create_dataloader(X_train, y_train, batch_size)
            val_dataloader = self.create_dataloader(X_val, y_val, batch_size)
            model = self.create_model(
                input_size=input_size,
                num_heads=num_heads,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
            self.train_model(model, train_dataloader, val_dataloader, learning_rate)
            accuracy = self.evaluate_model(model, val_dataloader)
            accuracies.append(accuracy)

        return np.mean(accuracies)

    def optimize(self, n_trials):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials)
        return study.best_params 