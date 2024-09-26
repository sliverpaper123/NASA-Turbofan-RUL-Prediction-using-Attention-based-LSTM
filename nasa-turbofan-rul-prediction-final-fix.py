import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns

# Hyperparameters
SEQUENCE_LENGTH = 50
HIDDEN_SIZE = 256
NUM_LAYERS = 2
DROPOUT_RATE = 0.3
LEARNING_RATE = 0.0001
BATCH_SIZE = 64
NUM_EPOCHS = 1000
EARLY_STOPPING_PATIENCE = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_data(train_file, test_file, rul_file):
    train_df = pd.read_csv(train_file, sep=' ', header=None)
    train_df.drop(columns=[26, 27], inplace=True)
    train_df.columns = ['unit', 'cycle'] + [f'op_setting_{i}' for i in range(1, 4)] + [f'sensor_{i}' for i in range(1, 22)]
    
    test_df = pd.read_csv(test_file, sep=' ', header=None)
    test_df.drop(columns=[26, 27], inplace=True)
    test_df.columns = train_df.columns
    
    rul_df = pd.read_csv(rul_file, sep=' ', header=None)
    rul_df = rul_df.iloc[:, 0].to_frame()
    rul_df.columns = ['rul']
    
    return train_df, test_df, rul_df

def add_features(df):
    # Calculate the change in sensor readings
    sensor_cols = [f'sensor_{i}' for i in range(1, 22)]
    for col in sensor_cols:
        df[f'{col}_change'] = df.groupby('unit')[col].diff().fillna(0)
    
    # Calculate moving averages
    window_sizes = [5, 10, 20]
    for col in sensor_cols:
        for window in window_sizes:
            df[f'{col}_ma_{window}'] = df.groupby('unit')[col].rolling(window=window, min_periods=1).mean().reset_index(0, drop=True)
    
    return df

def prepare_time_series_data(df, sequence_length, is_training=True):
    if is_training:
        rul = df.groupby('unit')['cycle'].transform('max') - df['cycle']
        df['rul'] = rul
    
    feature_cols = [col for col in df.columns if col not in ['unit', 'cycle', 'rul']]
    
    grouped = df.groupby('unit')
    sequences = []
    labels = []
    
    for _, group in grouped:
        if is_training:
            for i in range(0, len(group) - sequence_length + 1, sequence_length):
                seq = group[feature_cols].iloc[i:i+sequence_length].values
                if len(seq) == sequence_length:
                    sequences.append(seq)
                    labels.append(group['rul'].iloc[i+sequence_length-1])
        else:
            # For test data, use the last sequence_length cycles, or pad if necessary
            seq = group[feature_cols].iloc[-sequence_length:].values
            if len(seq) < sequence_length:
                pad_length = sequence_length - len(seq)
                seq = np.pad(seq, ((pad_length, 0), (0, 0)), mode='edge')
            sequences.append(seq)
            labels.append(group['cycle'].iloc[-1])
    
    return np.array(sequences), np.array(labels)

class AttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate):
        super(AttentionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.attention = nn.Linear(hidden_size, 1)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        out = self.fc1(self.dropout(context_vector))
        out = torch.relu(out)
        out = self.fc2(self.dropout(out))
        return out

def custom_loss(pred, true):
    diff = pred - true
    return torch.mean(torch.where(diff < 0, torch.abs(diff) * 1.2, torch.abs(diff)))

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience):
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y.unsqueeze(1))
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), 'best_model.pth', _use_new_zipfile_serialization=True)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping")
                break
    
    model.load_state_dict(torch.load('best_model.pth', map_location=device, weights_only=True))
    return model

def evaluate_predictions(true_rul, predicted_rul):
    rmse = np.sqrt(mean_squared_error(true_rul, predicted_rul))
    mae = mean_absolute_error(true_rul, predicted_rul)
    r2 = r2_score(true_rul, predicted_rul)
    return rmse, mae, r2

def plot_results(true_rul, predicted_rul, dataset):
    plt.figure(figsize=(12, 6))
    plt.scatter(true_rul, predicted_rul, alpha=0.5)
    plt.plot([true_rul.min(), true_rul.max()], [true_rul.min(), true_rul.max()], 'r--', lw=2)
    plt.xlabel('True RUL')
    plt.ylabel('Predicted RUL')
    plt.title(f'True vs Predicted RUL - Dataset {dataset}')
    plt.tight_layout()
    plt.savefig(f'rul_prediction_plot_{dataset}.png')
    plt.close()

    errors = predicted_rul - true_rul
    plt.figure(figsize=(12, 6))
    sns.histplot(errors, kde=True)
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Prediction Errors - Dataset {dataset}')
    plt.tight_layout()
    plt.savefig(f'error_distribution_{dataset}.png')
    plt.close()

def train_and_evaluate(train_file, test_file, rul_file, dataset):
    train_df, test_df, rul_df = load_data(train_file, test_file, rul_file)
    
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"RUL shape: {rul_df.shape}")
    
    train_df = add_features(train_df)
    test_df = add_features(test_df)
    
    X_train, y_train = prepare_time_series_data(train_df, SEQUENCE_LENGTH, is_training=True)
    X_test, _ = prepare_time_series_data(test_df, SEQUENCE_LENGTH, is_training=False)
    
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    
    rul_scaler = MinMaxScaler()
    y_train_scaled = rul_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    
    X_train_tensor, X_val_tensor, y_train_tensor, y_val_tensor = train_test_split(
        X_train_tensor, y_train_tensor, test_size=0.2, random_state=42
    )
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    input_size = X_train.shape[2]
    
    model = AttentionLSTM(input_size, HIDDEN_SIZE, NUM_LAYERS, DROPOUT_RATE).to(device)
    criterion = custom_loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    model = train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, EARLY_STOPPING_PATIENCE)
    
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test_tensor).cpu().numpy().flatten()
    
    test_predictions = rul_scaler.inverse_transform(test_predictions.reshape(-1, 1)).flatten()
    
    print(f"Test predictions shape: {test_predictions.shape}")
    print(f"RUL shape: {rul_df.shape}")
    
    rmse, mae, r2 = evaluate_predictions(rul_df['rul'], test_predictions)
    print(f"Dataset {dataset} Results:")
    print(f"Root Mean Square Error: {rmse:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"R-squared Score: {r2:.2f}")
    
    plot_results(rul_df['rul'], test_predictions, dataset)
    
    return test_predictions

# Main execution
datasets = ['FD001', 'FD002', 'FD003', 'FD004']

for dataset in datasets:
    print(f"\nProcessing dataset: {dataset}")
    train_file = f'train_{dataset}.txt'
    test_file = f'test_{dataset}.txt'
    rul_file = f'RUL_{dataset}.txt'
    
    try:
        predictions = train_and_evaluate(train_file, test_file, rul_file, dataset)
        np.savetxt(f'predictions_{dataset}.txt', predictions, fmt='%.2f')
        print(f"Predictions saved to predictions_{dataset}.txt")
    except Exception as e:
        print(f"Error processing dataset {dataset}: {str(e)}")
        import traceback
        traceback.print_exc()

print("\nAll datasets processed. Check the generated plot images for visual results.")
