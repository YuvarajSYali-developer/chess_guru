"""
CNN TRAINING DATA PREPARATION - FROM EXISTING CSV
==================================================

Standalone script to process your CSV and build ready-to-train data for the
Position Difficulty CNN.

Your CSV has: game_id, move_number, fen, cpl, eval_best, eval_played, phase, etc.

CRITICAL: We use CPL from CSV to CREATE LABELS, but NOT as features.
         The CNN will ONLY see the board position (from FEN).

This avoids the circular dependency from Phase 2.
"""

import numpy as np
import pandas as pd
import chess
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
import os

# ============================================
# HELPER FUNCTIONS
# ============================================

PIECE_TO_CHANNEL = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5
}

def fen_to_tensor(fen: str) -> np.ndarray:
    """
    Convert FEN to 8x8x14 tensor.
    
    Channels:
    0-5: White pieces (P,N,B,R,Q,K)
    6-11: Black pieces (P,N,B,R,Q,K)
    12: Side to move (1=current player)
    13: Castling rights (0-4)
    
    Always from current player's perspective.
    """
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


# ============================================
# DATA PREPARATION FROM CSV
# ============================================

def prepare_cnn_data_from_csv(
    csv_path,
    output_file='cnn_training_data.npz',
    output_csv=None,
    difficulty_threshold_high=50,  # CPL > 50 = difficult
    accuracy_threshold_low=20,     # CPL < 20 = easy
    skip_opening_moves=10,         # Skip first N moves (too theoretical)
    sample_rate=1.0,               # Sample fraction of eligible positions
    max_samples=None,              # Optional limit
    balance_classes=True,          # Balance easy/difficult samples
    random_state=42
):
    """
    Create CNN training data from your existing CSV.
    
    Your CSV columns:
    - fen: Board position
    - cpl: Centipawn loss (from Stockfish - already computed!)
    - phase_by_move: opening/middlegame/endgame
    - move_number: Move number
    - invalid_fen: Filter out invalid positions
    
    Key Principle:
    - CPL is used to CREATE LABELS (difficult vs easy)
    - But CNN features = ONLY board state from FEN
    - NO eval_best, eval_played, or CPL as features!
    
    Returns:
        X: numpy array of board tensors (N, 8, 8, 14)
        y: numpy array of labels (N,) - 0=easy, 1=difficult
        metadata: list of dicts with position info
    """
    
    print("ðŸ“‚ Loading CSV data...")
    df = pd.read_csv(csv_path)
    
    required_cols = ['fen', 'cpl']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")
    
    # Drop rows with missing essential fields
    df = df.dropna(subset=required_cols)
    
    print(f"âœ… Loaded {len(df):,} rows")
    print(f"   Columns: {df.columns.tolist()}")
    
    # ============================================
    # FILTER DATA
    # ============================================
    
    print("\nðŸ” Filtering data...")
    
    # Remove invalid positions
    if 'invalid_fen' in df.columns:
        df = df[df['invalid_fen'] == 0]
        print(f"   âœ“ Removed invalid FENs: {len(df):,} rows remaining")
    
    # Skip opening moves (too theoretical)
    if 'move_number' in df.columns:
        df = df[df['move_number'] > skip_opening_moves]
        print(f"   âœ“ Skipped first {skip_opening_moves} moves: {len(df):,} rows remaining")
    
    # Sample positions (reduce dataset size)
    if sample_rate < 1.0:
        df = df.sample(frac=sample_rate, random_state=random_state)
        print(f"   âœ“ Sampled {sample_rate*100:.0f}%: {len(df):,} rows remaining")
    
    # ============================================
    # CREATE LABELS FROM CPL
    # ============================================
    
    print("\nðŸ·ï¸  Creating labels from CPL...")
    
    df['cpl'] = pd.to_numeric(df['cpl'], errors='coerce')
    df = df.dropna(subset=['cpl'])
    
    # Categorize positions
    df['is_difficult'] = df['cpl'].apply(
        lambda cpl: 1 if cpl > difficulty_threshold_high 
                    else (0 if cpl < accuracy_threshold_low else -1)
    )
    
    # Remove ambiguous positions (CPL between thresholds)
    df = df[df['is_difficult'] != -1]
    
    n_difficult = (df['is_difficult'] == 1).sum()
    n_easy = (df['is_difficult'] == 0).sum()
    
    print(f"   âœ“ Difficult positions (CPL > {difficulty_threshold_high}): {n_difficult:,}")
    print(f"   âœ“ Easy positions (CPL < {accuracy_threshold_low}): {n_easy:,}")
    print(f"   âœ“ Total labeled: {len(df):,}")
    
    # ============================================
    # BALANCE CLASSES (OPTIONAL)
    # ============================================
    
    if balance_classes:
        print("\nâš–ï¸  Balancing classes...")
        
        df_difficult = df[df['is_difficult'] == 1]
        df_easy = df[df['is_difficult'] == 0]
        
        # Downsample majority class
        min_count = min(len(df_difficult), len(df_easy))
        
        df_difficult = df_difficult.sample(n=min_count, random_state=random_state)
        df_easy = df_easy.sample(n=min_count, random_state=random_state)
        
        df = pd.concat([df_difficult, df_easy], ignore_index=True)
        df = df.sample(frac=1.0, random_state=random_state)  # Shuffle
        
        print(f"   âœ“ Balanced to {min_count:,} samples per class")
        print(f"   âœ“ Total: {len(df):,} positions")
    
    # ============================================
    # LIMIT DATASET SIZE (OPTIONAL)
    # ============================================
    
    if max_samples and len(df) > max_samples:
        print(f"\nâœ‚ï¸  Limiting to {max_samples:,} samples...")
        df = df.sample(n=max_samples, random_state=random_state)
        print(f"   âœ“ Dataset size: {len(df):,}")
    
    # ============================================
    # CONVERT TO TENSORS
    # ============================================
    
    print("\nðŸ”„ Converting positions to tensors...")
    print("   This may take a few minutes for large datasets...")
    
    X_list = []
    y_list = []
    metadata_list = []
    
    # Progress bar
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        try:
            # Convert FEN to tensor (NO eval features!)
            tensor = fen_to_tensor(row['fen'])
            
            # Get label
            label = row['is_difficult']
            
            # Store
            X_list.append(tensor)
            y_list.append(label)
            
            # Metadata (for analysis, not for training)
            metadata_list.append({
                'game_id': row.get('game_id', None),
                'move_number': row.get('move_number', None),
                'cpl': row.get('cpl', None),
                'phase': row.get('phase_by_move', None),
                'side': row.get('side', None)
            })
            
        except Exception as e:
            # Skip positions that can't be parsed
            continue
    
    # Convert to numpy arrays
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    
    # ============================================
    # SUMMARY STATISTICS
    # ============================================
    
    print("\n" + "="*60)
    print("ðŸ“Š DATASET SUMMARY")
    print("="*60)
    print(f"Total positions: {len(X):,}")
    print(f"Feature shape: {X.shape}")
    print(f"Label shape: {y.shape}")
    print(f"\nClass distribution:")
    print(f"  Easy (0): {(y==0).sum():,} ({(y==0).sum()/len(y)*100:.1f}%)")
    print(f"  Difficult (1): {(y==1).sum():,} ({(y==1).sum()/len(y)*100:.1f}%)")
    
    # Phase distribution (if available)
    if metadata_list and metadata_list[0]['phase']:
        phase_counts = {}
        for meta in metadata_list:
            phase = meta['phase']
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
        
        print(f"\nPhase distribution:")
        for phase, count in phase_counts.items():
            print(f"  {phase}: {count:,} ({count/len(metadata_list)*100:.1f}%)")
    
    print(f"\nMemory usage: {X.nbytes / 1e9:.2f} GB")
    print("="*60)
    
    # ============================================
    # SAVE TO DISK
    # ============================================

    if output_csv:
        print(f"\nSaving labeled CSV to {output_csv}...")
        df.to_csv(output_csv, index=False)
        print("CSV saved!")
    
    print(f"\nðŸ’¾ Saving to {output_file}...")
    
    np.savez_compressed(
        output_file,
        X=X,
        y=y,
        metadata=metadata_list
    )
    
    print("âœ… Data saved!")
    print(f"\nTo load:")
    print(f"  data = np.load('{output_file}', allow_pickle=True)")
    print(f"  X = data['X']")
    print(f"  y = data['y']")
    print(f"  metadata = data['metadata']")
    
    return X, y, metadata_list


# ============================================
# TRAIN/VAL/TEST SPLIT
# ============================================

def create_splits(X, y, metadata, 
                  test_size=0.2, 
                  val_size=0.15,
                  output_dir='./data',
                  random_state=42):
    """
    Split data into train/val/test sets and save separately.
    
    This is useful for:
    - Training in different Colab sessions
    - Loading only what you need
    - Easier sharing
    """
    
    print("\nâœ‚ï¸  Creating train/val/test splits...")
    
    # First split: separate test set
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: separate validation from training
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, 
        test_size=val_ratio, 
        random_state=random_state, 
        stratify=y_trainval
    )
    
    print(f"  Train: {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Val:   {len(X_val):,} ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test:  {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)")
    
    # Save splits
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    np.savez_compressed(f'{output_dir}/train.npz', X=X_train, y=y_train)
    np.savez_compressed(f'{output_dir}/val.npz', X=X_val, y=y_val)
    np.savez_compressed(f'{output_dir}/test.npz', X=X_test, y=y_test)
    
    print(f"\nâœ… Splits saved to {output_dir}/")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================
# DATA ANALYSIS (OPTIONAL)
# ============================================

def analyze_dataset(df, y, metadata):
    """
    Analyze the prepared dataset to understand patterns.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    print("\nðŸ“Š Dataset Analysis")
    print("="*60)
    
    # CPL distribution by label
    difficult_cpls = [m['cpl'] for i, m in enumerate(metadata) if y[i] == 1]
    easy_cpls = [m['cpl'] for i, m in enumerate(metadata) if y[i] == 0]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # CPL distribution
    axes[0].hist(difficult_cpls, bins=50, alpha=0.7, label='Difficult', color='red')
    axes[0].hist(easy_cpls, bins=50, alpha=0.7, label='Easy', color='green')
    axes[0].set_xlabel('CPL')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('CPL Distribution by Difficulty')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Phase distribution by label
    phases_difficult = [m['phase'] for i, m in enumerate(metadata) if y[i] == 1 and m['phase']]
    phases_easy = [m['phase'] for i, m in enumerate(metadata) if y[i] == 0 and m['phase']]
    
    phase_counts_diff = pd.Series(phases_difficult).value_counts()
    phase_counts_easy = pd.Series(phases_easy).value_counts()
    
    x = np.arange(len(phase_counts_diff))
    width = 0.35
    
    axes[1].bar(x - width/2, phase_counts_diff.values, width, label='Difficult', color='red', alpha=0.7)
    axes[1].bar(x + width/2, phase_counts_easy.values, width, label='Easy', color='green', alpha=0.7)
    axes[1].set_xlabel('Game Phase')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Phase Distribution by Difficulty')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(phase_counts_diff.index)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('dataset_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("âœ… Analysis plot saved to dataset_analysis.png")


# ============================================
# CLI
# ============================================


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description="Prepare CNN training data from a CSV (FEN + CPL)."
    )
    p.add_argument("--csv", required=True, help="Path to input CSV.")
    p.add_argument("--out", default="cnn_training_data.npz",
                   help="Output .npz file for full dataset.")
    p.add_argument("--out-csv", default=None,
                   help="Optional output CSV of filtered/labeled data.")
    p.add_argument("--difficulty-high", type=float, default=50.0,
                   help="CPL > this => difficult.")
    p.add_argument("--accuracy-low", type=float, default=20.0,
                   help="CPL < this => easy.")
    p.add_argument("--skip-opening-moves", type=int, default=10,
                   help="Skip first N moves.")
    p.add_argument("--sample-rate", type=float, default=1.0,
                   help="Fraction of positions to sample (0-1].")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Optional cap on total samples.")
    p.add_argument("--no-balance", action="store_true",
                   help="Disable class balancing.")
    p.add_argument("--split-dir", default=None,
                   help="If set, save train/val/test splits here.")
    p.add_argument("--test-size", type=float, default=0.2,
                   help="Test split fraction.")
    p.add_argument("--val-size", type=float, default=0.15,
                   help="Validation split fraction.")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for sampling/balancing/splits.")
    return p


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    print("\n" + "="*60)
    print("CNN DATA PREPARATION FROM CSV")
    print("="*60)
    print("Key Points:")
    print("  - Uses CPL to create labels (difficult vs easy)")
    print("  - CNN features = ONLY board state from FEN")
    print("  - NO eval_best, eval_played, or CPL as features")
    print("  - Avoids circular dependency from Phase 2")
    print("="*60 + "\n")

    X, y, metadata = prepare_cnn_data_from_csv(
        csv_path=args.csv,
        output_file=args.out,
        output_csv=args.out_csv,
        difficulty_threshold_high=args.difficulty_high,
        accuracy_threshold_low=args.accuracy_low,
        skip_opening_moves=args.skip_opening_moves,
        sample_rate=args.sample_rate,
        max_samples=args.max_samples,
        balance_classes=not args.no_balance,
        random_state=args.seed
    )

    if args.split_dir:
        os.makedirs(args.split_dir, exist_ok=True)
        create_splits(
            X, y, metadata,
            test_size=args.test_size,
            val_size=args.val_size,
            output_dir=args.split_dir,
            random_state=args.seed
        )
