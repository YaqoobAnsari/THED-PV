"""
Dataset Splitting for HR-ThermalPV
Splits homography pairs into train/val/test sets with stratification
"""

import numpy as np
from pathlib import Path
import argparse
import json
import shutil
from tqdm import tqdm
import logging
from typing import List, Dict, Tuple
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatasetSplitter:
    """Split homography pairs into train/val/test sets"""
    
    def __init__(
        self,
        train_ratio: float = 0.5,
        val_ratio: float = 0.25,
        test_ratio: float = 0.25,
        seed: int = 42
    ):
        """
        Args:
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            seed: Random seed for reproducibility
        """
        # Validate ratios
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError("Ratios must sum to 1.0")
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        
        np.random.seed(seed)
        
        logger.info(f"Dataset splitter initialized:")
        logger.info(f"  Train: {train_ratio:.1%}")
        logger.info(f"  Val:   {val_ratio:.1%}")
        logger.info(f"  Test:  {test_ratio:.1%}")
        logger.info(f"  Seed:  {seed}")
    
    def get_sample_metadata(self, sample_dir: Path) -> Dict:
        """Extract metadata from sample directory"""
        metadata_path = sample_dir / "metadata.json"
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        else:
            # Basic metadata if file missing
            return {
                'sample_id': sample_dir.name,
                'exists': False
            }
    
    def stratified_split(
        self,
        samples: List[Path]
    ) -> Tuple[List[Path], List[Path], List[Path]]:
        """Split samples ensuring temporal/condition stratification"""
        
        # Group samples by parent directory (preserves day/time structure)
        groups = defaultdict(list)
        for sample in samples:
            # Get parent path up to day level
            # e.g., .../2024-12-22/8am/10cm -> group by 2024-12-22/8am
            parts = sample.parts
            if len(parts) >= 3:
                group_key = '/'.join(parts[-3:-1])  # Day/Time grouping
            else:
                group_key = 'unknown'
            groups[group_key].append(sample)
        
        logger.info(f"Found {len(groups)} temporal groups for stratification")
        
        train_samples = []
        val_samples = []
        test_samples = []
        
        # Split each group proportionally
        for group_key, group_samples in groups.items():
            n = len(group_samples)
            
            # Shuffle within group
            shuffled = np.random.permutation(group_samples)
            
            # Calculate split points
            train_end = int(n * self.train_ratio)
            val_end = train_end + int(n * self.val_ratio)
            
            train_samples.extend(shuffled[:train_end])
            val_samples.extend(shuffled[train_end:val_end])
            test_samples.extend(shuffled[val_end:])
        
        return train_samples, val_samples, test_samples
    
    def copy_sample(self, src_dir: Path, dst_dir: Path):
        """Copy sample directory to destination"""
        dst_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy all files in sample directory
        for file in src_dir.iterdir():
            if file.is_file():
                shutil.copy2(file, dst_dir / file.name)
    
    def create_split_summary(
        self,
        output_dir: Path,
        train: List[Path],
        val: List[Path],
        test: List[Path]
    ):
        """Create summary statistics for splits"""
        
        summary = {
            'seed': self.seed,
            'ratios': {
                'train': self.train_ratio,
                'val': self.val_ratio,
                'test': self.test_ratio
            },
            'counts': {
                'train': len(train),
                'val': len(val),
                'test': len(test),
                'total': len(train) + len(val) + len(test)
            },
            'actual_ratios': {
                'train': len(train) / (len(train) + len(val) + len(test)),
                'val': len(val) / (len(train) + len(val) + len(test)),
                'test': len(test) / (len(train) + len(val) + len(test))
            }
        }
        
        # Save summary
        with open(output_dir / "split_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save sample lists
        for split_name, split_samples in [('train', train), ('val', val), ('test', test)]:
            with open(output_dir / f"{split_name}_samples.txt", 'w') as f:
                for sample in sorted(split_samples):
                    f.write(f"{sample.name}\n")
        
        return summary


def split_dataset(
    input_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.5,
    val_ratio: float = 0.25,
    test_ratio: float = 0.25,
    seed: int = 42,
    use_symlinks: bool = False
):
    """Split homography pairs dataset"""
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    # Create output directories
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    test_dir = output_dir / "test"
    
    for d in [train_dir, val_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Initialize splitter
    splitter = DatasetSplitter(train_ratio, val_ratio, test_ratio, seed)
    
    # Find all sample directories
    sample_dirs = [d for d in input_dir.iterdir() if d.is_dir() and d.name.startswith('sample_')]
    
    if not sample_dirs:
        raise ValueError(f"No sample directories found in {input_dir}")
    
    logger.info(f"Found {len(sample_dirs)} samples")
    
    # Stratified split
    train_samples, val_samples, test_samples = splitter.stratified_split(sample_dirs)
    
    logger.info(f"\nSplit results:")
    logger.info(f"  Train: {len(train_samples)} samples ({len(train_samples)/len(sample_dirs):.1%})")
    logger.info(f"  Val:   {len(val_samples)} samples ({len(val_samples)/len(sample_dirs):.1%})")
    logger.info(f"  Test:  {len(test_samples)} samples ({len(test_samples)/len(sample_dirs):.1%})")
    
    # Copy/link samples to split directories
    logger.info("\nCopying samples...")
    
    for sample in tqdm(train_samples, desc="Train"):
        splitter.copy_sample(sample, train_dir / sample.name)
    
    for sample in tqdm(val_samples, desc="Val"):
        splitter.copy_sample(sample, val_dir / sample.name)
    
    for sample in tqdm(test_samples, desc="Test"):
        splitter.copy_sample(sample, test_dir / sample.name)
    
    # Create summary
    summary = splitter.create_split_summary(
        output_dir, train_samples, val_samples, test_samples
    )
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Dataset split complete!")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Summary saved to: {output_dir / 'split_summary.json'}")
    logger.info(f"{'='*60}\n")
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Split HR-ThermalPV dataset into train/val/test"
    )
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory with homography pairs')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for splits')
    parser.add_argument('--train_ratio', type=float, default=0.5,
                        help='Training set ratio (default: 0.5)')
    parser.add_argument('--val_ratio', type=float, default=0.25,
                        help='Validation set ratio (default: 0.25)')
    parser.add_argument('--test_ratio', type=float, default=0.25,
                        help='Test set ratio (default: 0.25)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--symlinks', action='store_true',
                        help='Use symlinks instead of copying (faster, less space)')
    
    args = parser.parse_args()
    
    split_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        use_symlinks=args.symlinks
    )


if __name__ == "__main__":
    main()