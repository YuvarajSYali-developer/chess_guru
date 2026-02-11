# ============================================
# GOOGLE COLAB NOTEBOOK: CNN POSITION DIFFICULTY
# Complete Pipeline from CSV to Trained Model
# ============================================

# Copy this entire notebook to Google Colab
# Run cells in order

# ===================================================
# CELL 1: Setup & Mount Drive
# ===================================================

from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import pandas as pd
import chess
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
import pickle
import json

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

# Create project directory
import os
base_path = '/content/drive/MyDrive/chess_project'
os.makedirs(base_path, exist_ok=True)
os.makedirs(f'{base_path}/data', exist_ok=True)
os.makedirs(f'{base_path}/models', exist_ok=True)
os.makedirs(f'{base_path}/results', exist_ok=True)

print("✅ Setup complete!")


# ===================================================
# CELL 2: Define Helper Functions
# ===================================================

PIECE_TO_CHANNEL = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5
}

def fen_to_tensor(fen: str) -> np.ndarray:
    """Convert FEN to 8x8x14 tensor (ONLY board state, no evals!)"""
    board = chess.Board(fen)
    tensor = np.zeros((8, 8, 14), dtype=np.float32)
    
    # Piece placement
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
        
        row = 7 - chess.square_rank(square)
        col = chess.square_file(square)
        base_channel = PIECE_TO_CHANNEL[piece.piece_type]
        channel = base_channel if piece.color == chess.WHITE else base_channel + 6
        tensor[row, col, channel] = 1.0
    
    # Side to move
    stm_value = 1.0 if board.turn == chess.WHITE else 0.0
    tensor[:, :, 12] = stm_value
    
    # Castling rights
    castling_count = sum([
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK)
    ])
    tensor[:, :, 13] = float(castling_count)
    
    # Normalize perspective
    if board.turn == chess.BLACK:
        tensor = np.flip(tensor, axis=0)
        tensor[:, :, 0:6], tensor[:, :, 6:12] = (
            tensor[:, :, 6:12].copy(),
            tensor[:, :, 0:6].copy()
        )
        tensor[:, :, 12] = 1.0
    
    return tensor

print("✅ Helper functions defined")


# ===================================================
# CELL 3: Load and Process CSV Data
# ===================================================

# ADJUST THESE PATHS TO YOUR FILES
csv_path = f'{base_path}/your_chess_data.csv'  # ← CHANGE THIS

print("📂 Loading CSV...")
df = pd.read_csv(csv_path)

print(f"✅ Loaded {len(df):,} rows")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())


# ===================================================
# CELL 4: Filter and Prepare Data
# ===================================================

print("\n🔍 Filtering data...")

# Remove invalid FENs
if 'invalid_fen' in df.columns:
    df = df[df['invalid_fen'] == 0]
    print(f"  ✓ Removed invalid FENs: {len(df):,} remaining")

# Skip opening moves (first 10 moves are too theoretical)
if 'move_number' in df.columns:
    df = df[df['move_number'] > 10]
    print(f"  ✓ Skipped opening moves: {len(df):,} remaining")

# Sample data (adjust based on available RAM)
# Start with 20% for testing, increase later
SAMPLE_RATE = 0.2  # ← ADJUST THIS (0.1 = 10%, 0.5 = 50%, 1.0 = 100%)

df = df.sample(frac=SAMPLE_RATE, random_state=42)
print(f"  ✓ Sampled {SAMPLE_RATE*100:.0f}%: {len(df):,} rows")

print("\n🏷️  Creating labels from CPL...")

# CRITICAL: Use CPL to create labels, NOT as features!
# Difficult = CPL > 50 (human made mistake)
# Easy = CPL < 20 (human played well)

DIFFICULTY_THRESHOLD = 50  # ← ADJUST THIS
ACCURACY_THRESHOLD = 20    # ← ADJUST THIS

def categorize_position(cpl):
    if cpl > DIFFICULTY_THRESHOLD:
        return 1  # Difficult
    elif cpl < ACCURACY_THRESHOLD:
        return 0  # Easy
    else:
        return -1  # Ambiguous (skip)

df['is_difficult'] = df['cpl'].apply(categorize_position)
df = df[df['is_difficult'] != -1]  # Remove ambiguous

n_difficult = (df['is_difficult'] == 1).sum()
n_easy = (df['is_difficult'] == 0).sum()

print(f"  ✓ Difficult positions: {n_difficult:,}")
print(f"  ✓ Easy positions: {n_easy:,}")

# Balance classes
print("\n⚖️  Balancing classes...")
df_difficult = df[df['is_difficult'] == 1]
df_easy = df[df['is_difficult'] == 0]

min_count = min(len(df_difficult), len(df_easy))
df_difficult = df_difficult.sample(n=min_count, random_state=42)
df_easy = df_easy.sample(n=min_count, random_state=42)

df = pd.concat([df_difficult, df_easy], ignore_index=True)
df = df.sample(frac=1.0, random_state=42)  # Shuffle

print(f"  ✓ Balanced to {min_count:,} per class")
print(f"  ✓ Total: {len(df):,} positions")


# ===================================================
# CELL 5: Convert to Tensors
# ===================================================

print("\n🔄 Converting FENs to tensors...")
print("This may take a few minutes...")

X_list = []
y_list = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    try:
        # Convert FEN to tensor (NO eval features!)
        tensor = fen_to_tensor(row['fen'])
        label = row['is_difficult']
        
        X_list.append(tensor)
        y_list.append(label)
    except:
        continue  # Skip problematic positions

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.int32)

print(f"\n✅ Created {len(X):,} position tensors")
print(f"   Shape: {X.shape}")
print(f"   Memory: {X.nbytes / 1e9:.2f} GB")
print(f"   Difficult: {(y==1).sum():,} ({(y==1).sum()/len(y)*100:.1f}%)")
print(f"   Easy: {(y==0).sum():,} ({(y==0).sum()/len(y)*100:.1f}%)")


# ===================================================
# CELL 6: Train/Val/Test Split
# ===================================================

print("\n✂️  Creating splits...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
)

print(f"Train: {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)")
print(f"Val:   {len(X_val):,} ({len(X_val)/len(X)*100:.1f}%)")
print(f"Test:  {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)")

# Save splits
np.savez_compressed(f'{base_path}/data/train.npz', X=X_train, y=y_train)
np.savez_compressed(f'{base_path}/data/val.npz', X=X_val, y=y_val)
np.savez_compressed(f'{base_path}/data/test.npz', X=X_test, y=y_test)

print(f"\n✅ Splits saved to {base_path}/data/")


# ===================================================
# CELL 7: Build CNN Model
# ===================================================

print("\n🧠 Building CNN model...")

model = Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', padding='same', 
                 input_shape=(8, 8, 14), name='conv1'),
    layers.BatchNormalization(),
    
    layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(256, (2, 2), activation='relu', padding='same', name='conv4'),
    layers.BatchNormalization(),
    
    layers.GlobalAveragePooling2D(),
    
    layers.Dense(512, activation='relu', name='fc1'),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu', name='fc2'),
    layers.Dropout(0.4),
    layers.Dense(128, activation='relu', name='fc3'),
    layers.Dropout(0.3),
    
    layers.Dense(1, activation='sigmoid', name='output')
], name='PositionDifficultyCNN')

model.summary()

# Compile
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        keras.metrics.AUC(name='auc'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall')
    ]
)

print("✅ Model compiled")


# ===================================================
# CELL 8: Train Model
# ===================================================

print("\n🚀 Training model...")

callbacks = [
    keras.callbacks.ModelCheckpoint(
        f'{base_path}/models/best_model.h5',
        monitor='val_auc',
        mode='max',
        save_best_only=True,
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
]

history = model.fit(
    X_train, y_train,
    batch_size=128,
    epochs=50,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)

print("\n✅ Training complete!")


# ===================================================
# CELL 9: Evaluate Model
# ===================================================

print("\n📊 Evaluating on test set...")

test_loss, test_acc, test_auc, test_prec, test_rec = model.evaluate(
    X_test, y_test, verbose=0
)

y_pred_prob = model.predict(X_test, verbose=0).flatten()
y_pred = (y_pred_prob > 0.5).astype(int)

print("\n" + "="*60)
print("TEST SET RESULTS")
print("="*60)
print(f"Loss: {test_loss:.4f}")
print(f"Accuracy: {test_acc:.4f}")
print(f"AUC: {test_auc:.4f}")
print(f"Precision: {test_prec:.4f}")
print(f"Recall: {test_rec:.4f}")
print("="*60)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Easy', 'Difficult']))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Save results
results = {
    'test_loss': float(test_loss),
    'test_accuracy': float(test_acc),
    'test_auc': float(test_auc),
    'test_precision': float(test_prec),
    'test_recall': float(test_rec),
    'confusion_matrix': cm.tolist()
}

with open(f'{base_path}/results/test_results.json', 'w') as f:
    json.dump(results, f, indent=2)


# ===================================================
# CELL 10: Plot Results
# ===================================================

# Training history
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Loss
axes[0, 0].plot(history.history['loss'], label='Train')
axes[0, 0].plot(history.history['val_loss'], label='Val')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training & Validation Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Accuracy
axes[0, 1].plot(history.history['accuracy'], label='Train')
axes[0, 1].plot(history.history['val_accuracy'], label='Val')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].set_title('Training & Validation Accuracy')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# AUC
axes[1, 0].plot(history.history['auc'], label='Train')
axes[1, 0].plot(history.history['val_auc'], label='Val')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('AUC')
axes[1, 0].set_title('Training & Validation AUC')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1],
            xticklabels=['Easy', 'Difficult'],
            yticklabels=['Easy', 'Difficult'])
axes[1, 1].set_ylabel('True Label')
axes[1, 1].set_xlabel('Predicted Label')
axes[1, 1].set_title('Confusion Matrix')

plt.tight_layout()
plt.savefig(f'{base_path}/results/training_results.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n✅ Plot saved to {base_path}/results/training_results.png")


# ===================================================
# CELL 11: Test on Sample Positions
# ===================================================

print("\n🧪 Testing on sample positions...")

# Test a few positions
sample_fens = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # Starting position
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",  # Italian
    "8/8/4k3/8/8/4K3/4P3/8 w - - 0 1",  # Simple endgame
]

for fen in sample_fens:
    tensor = fen_to_tensor(fen).reshape(1, 8, 8, 14)
    difficulty_score = model.predict(tensor, verbose=0)[0][0]
    
    print(f"\nFEN: {fen}")
    print(f"Difficulty Score: {difficulty_score:.3f}")
    if difficulty_score > 0.7:
        print("→ Very Difficult (high chance of mistakes)")
    elif difficulty_score > 0.5:
        print("→ Challenging")
    elif difficulty_score > 0.3:
        print("→ Moderate")
    else:
        print("→ Easy (straightforward)")


# ===================================================
# CELL 12: Save Everything
# ===================================================

print("\n💾 Saving final model and metadata...")

# Save model
model.save(f'{base_path}/models/final_model.h5')

# Save training history
with open(f'{base_path}/results/history.pkl', 'wb') as f:
    pickle.dump(history.history, f)

# Save configuration
config = {
    'sample_rate': SAMPLE_RATE,
    'difficulty_threshold': DIFFICULTY_THRESHOLD,
    'accuracy_threshold': ACCURACY_THRESHOLD,
    'total_positions': len(X),
    'train_samples': len(X_train),
    'val_samples': len(X_val),
    'test_samples': len(X_test),
    'test_accuracy': float(test_acc),
    'test_auc': float(test_auc)
}

with open(f'{base_path}/results/config.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f"""
{'='*60}
✅ ALL DONE! 
{'='*60}

Your files are saved in: {base_path}

Models:
  - {base_path}/models/best_model.h5
  - {base_path}/models/final_model.h5

Results:
  - {base_path}/results/test_results.json
  - {base_path}/results/training_results.png
  - {base_path}/results/config.json

Data:
  - {base_path}/data/train.npz
  - {base_path}/data/val.npz
  - {base_path}/data/test.npz

Next Steps:
1. Download the best_model.h5 to use in your Streamlit app
2. Use this model to predict position difficulty
3. Combine with Stockfish CPL for complete analysis

{'='*60}
""")


# ===================================================
# BONUS CELL: Download Model
# ===================================================

# Uncomment to download model to your computer
# from google.colab import files
# files.download(f'{base_path}/models/best_model.h5')
