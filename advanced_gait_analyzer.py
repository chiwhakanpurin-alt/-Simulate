import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# TensorFlow/Keras for LSTM
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_TF = True
except:
    HAS_TF = False
    print("⚠️  TensorFlow not available, using simpler model")

class AdvancedGaitAnalyzer:
    """Advanced time-series analysis of running gait data using LSTM"""
    
    def __init__(self, sequence_length=50):
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        
    def generate_grf_sequences(self, n_samples=500):
        """Generate time-series GRF data"""
        X_data = []
        y_data = []
        
        for label in [0, 1, 2]:  # Normal, Overpronate, Injured
            n = int(n_samples / 3)
            
            for _ in range(n):
                if label == 0:  # Normal
                    # Smooth, symmetric GRF profile
                    t = np.linspace(0, 2*np.pi, self.sequence_length)
                    grf = 2.5 + 1.0 * np.sin(t) + 0.8 * np.sin(2*t)
                    grf += np.random.normal(0, 0.05, self.sequence_length)
                    
                elif label == 1:  # Overpronate
                    # Higher impact, asymmetric
                    t = np.linspace(0, 2*np.pi, self.sequence_length)
                    grf = 3.0 + 1.3 * np.sin(t) + 0.6 * np.sin(2*t) + 0.3 * np.sin(3*t)
                    grf += np.random.normal(0, 0.08, self.sequence_length)
                    
                else:  # Injured
                    # Irregular, unstable pattern
                    t = np.linspace(0, 2*np.pi, self.sequence_length)
                    grf = 2.3 + 0.9 * np.sin(t) + 0.5 * np.sin(2*t)
                    grf[self.sequence_length//2:] *= 0.8  # Reduced load
                    grf += np.random.normal(0, 0.12, self.sequence_length)
                
                grf = np.maximum(grf, 0)  # Force cannot be negative
                X_data.append(grf)
                y_data.append(label)
        
        return np.array(X_data), np.array(y_data)
    
    def prepare_data(self, X, y, test_size=0.2):
        """Prepare data for LSTM"""
        # Reshape for LSTM [samples, timesteps, features]
        X_reshaped = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Split data
        split_idx = int(len(X) * (1 - test_size))
        X_train = X_reshaped[:split_idx]
        X_test = X_reshaped[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        # Scale
        X_train_flat = X_train.reshape(-1, 1)
        X_train_flat = self.scaler.fit_transform(X_train_flat)
        X_train = X_train_flat.reshape(X_train.shape)
        
        X_test_flat = X_test.reshape(-1, 1)
        X_test_flat = self.scaler.transform(X_test_flat)
        X_test = X_test_flat.reshape(X_test.shape)
        
        return X_train, X_test, y_train, y_test
    
    def build_lstm_model(self, input_shape):
        """Build LSTM model"""
        model = Sequential([
            LSTM(64, activation='relu', input_shape=input_shape, return_sequences=True),
            Dropout(0.2),
            LSTM(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(3, activation='softmax')  # 3 classes
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, X_train, X_test, y_train, y_test):
        """Train LSTM model"""
        model = self.build_lstm_model(X_train.shape[1:])
        
        print("🧠 Training LSTM Model...")
        history = model.fit(
            X_train, y_train,
            epochs=30,
            batch_size=16,
            validation_split=0.2,
            callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
            verbose=0
        )
        
        # Evaluate
        train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        
        print(f"\n📊 LSTM Model Performance:")
        print(f"  Train Accuracy: {train_acc:.3f}")
        print(f"  Test Accuracy: {test_acc:.3f}")
        
        # Predictions
        y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
        
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['Normal', 'Overpronate', 'Injured']))
        
        return model, history, y_pred

def simple_comparison_model():
    """Fallback: Simple statistical comparison without TF"""
    print("\n⚙️  Running Statistical Gait Analysis (No Deep Learning)...")
    
    # Generate simple sequences
    analyzer = AdvancedGaitAnalyzer()
    X, y = analyzer.generate_grf_sequences(300)
    
    # Extract statistics
    stats = pd.DataFrame({
        'mean_force': np.mean(X, axis=1),
        'max_force': np.max(X, axis=1),
        'std_force': np.std(X, axis=1),
        'range_force': np.max(X, axis=1) - np.min(X, axis=1),
        'label': y
    })
    
    print("\n📊 GRF Statistics by Gait Type:")
    for label in [0, 1, 2]:
        label_name = ['Normal', 'Overpronate', 'Injured'][label]
        subset = stats[stats['label'] == label]
        print(f"\n{label_name}:")
        print(f"  Mean Force: {subset['mean_force'].mean():.2f} ± {subset['mean_force'].std():.2f}")
        print(f"  Max Force:  {subset['max_force'].mean():.2f} ± {subset['max_force'].std():.2f}")
        print(f"  Std Dev:    {subset['std_force'].mean():.2f} ± {subset['std_force'].std():.2f}")
    
    # Save
    stats.to_csv('/mnt/user-data/outputs/gait_statistics.csv', index=False)
    return stats

def visualize_grf_sequences():
    """Visualize GRF time series"""
    analyzer = AdvancedGaitAnalyzer(sequence_length=100)
    X, y = analyzer.generate_grf_sequences(30)
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    labels = ['Normal', 'Overpronate', 'Injured']
    
    for idx, label in enumerate([0, 1, 2]):
        sample_idx = np.where(y == label)[0][0]
        
        axes[idx].plot(X[sample_idx], linewidth=2, color=['green', 'orange', 'red'][idx])
        axes[idx].fill_between(range(len(X[sample_idx])), X[sample_idx], alpha=0.3)
        axes[idx].set_title(f'{labels[idx]} - Ground Reaction Force Pattern', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Force (Body Weights)')
        axes[idx].grid(True, alpha=0.3)
        
        if idx == 2:
            axes[idx].set_xlabel('Time (% Gait Cycle)')
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/grf_time_series.png', dpi=300)
    print("\n📈 GRF Time Series saved: grf_time_series.png")
    plt.close()

# ============= RUN ANALYSIS =============
if __name__ == "__main__":
    print("🏃 Advanced Gait Analysis System\n")
    print("=" * 50)
    
    # Visualize sequences first
    print("📊 Generating GRF Time Series...")
    visualize_grf_sequences()
    
    if HAS_TF:
        print("\n" + "=" * 50)
        print("🤖 Advanced LSTM Analysis")
        print("=" * 50)
        
        analyzer = AdvancedGaitAnalyzer(sequence_length=50)
        X, y = analyzer.generate_grf_sequences(n_samples=600)
        X_train, X_test, y_train, y_test = analyzer.prepare_data(X, y)
        
        model, history, y_pred = analyzer.train_model(X_train, X_test, y_train, y_test)
        
        # Plot training history
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(history.history['accuracy'], label='Train Accuracy')
        ax.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('LSTM Model Training History')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/lstm_training_history.png', dpi=300)
        print("📈 Training history saved: lstm_training_history.png")
        plt.close()
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(['Normal', 'Overpronate', 'Injured'])
        ax.set_yticklabels(['Normal', 'Overpronate', 'Injured'])
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        ax.set_title('Confusion Matrix - LSTM Model')
        
        for i in range(3):
            for j in range(3):
                text = ax.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=14)
        
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/lstm_confusion_matrix.png', dpi=300)
        print("📊 Confusion matrix saved: lstm_confusion_matrix.png")
        plt.close()
        
    else:
        print("\n" + "=" * 50)
        stats = simple_comparison_model()
        print("\n📊 Statistical analysis saved: gait_statistics.csv")
    
    print("\n✅ Analysis Complete!")
