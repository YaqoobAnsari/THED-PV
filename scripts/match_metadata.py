#!/usr/bin/env python3
"""
match_metadata.py - Automatic alignment of thermal images with environmental measurements

This script automatically matches thermal images with environmental metadata based on
timestamp alignment. It handles the temporal synchronization between thermal image
capture times and environmental sensor readings (1-minute resolution).

Author: THED-PV Team
Date: January 2025
Python: 3.8+
Dependencies: pandas, numpy
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class MetadataMatcher:
    """
    Matches thermal images with environmental metadata based on timestamps.
    
    Environmental data is recorded at 1-minute resolution, while thermal images
    may be captured at slightly different times. This class handles the temporal
    alignment using configurable matching strategies.
    """
    
    def __init__(self, metadata_dir: Union[str, Path], 
                 time_tolerance_seconds: int = 60,
                 interpolation_method: str = 'nearest'):
        """
        Initialize the MetadataMatcher.
        
        Args:
            metadata_dir: Directory containing environmental metadata CSV files
            time_tolerance_seconds: Maximum time difference for matching (default: 60s)
            interpolation_method: Method for temporal alignment ('nearest', 'linear', 'forward', 'backward')
        """
        self.metadata_dir = Path(metadata_dir)
        self.time_tolerance = timedelta(seconds=time_tolerance_seconds)
        self.interpolation_method = interpolation_method
        
        # Storage for loaded metadata
        self.metadata_cache = {}
        self.date_to_file_map = {}
        
        # Load all metadata files
        self._load_metadata_files()
    
    def _load_metadata_files(self):
        """Load all environmental metadata CSV files into memory."""
        if not self.metadata_dir.exists():
            raise FileNotFoundError(f"Metadata directory not found: {self.metadata_dir}")
        
        # Find all CSV files
        csv_files = list(self.metadata_dir.glob("environmental_day*.csv"))
        
        if not csv_files:
            csv_files = list(self.metadata_dir.glob("*.csv"))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.metadata_dir}")
        
        print(f"Loading {len(csv_files)} metadata file(s)...")
        
        for csv_file in sorted(csv_files):
            try:
                # Load CSV
                df = pd.read_csv(csv_file)
                
                # Parse timestamp column (try common column names)
                timestamp_col = self._find_timestamp_column(df)
                
                if timestamp_col is None:
                    print(f"Warning: No timestamp column found in {csv_file.name}, skipping...")
                    continue
                
                # Convert to datetime
                df['timestamp'] = pd.to_datetime(df[timestamp_col])
                df = df.sort_values('timestamp')
                df = df.set_index('timestamp')
                
                # Extract date for quick lookup
                dates = df.index.date
                unique_dates = np.unique(dates)
                
                # Store in cache
                for date in unique_dates:
                    date_str = date.strftime('%Y-%m-%d')
                    self.date_to_file_map[date_str] = csv_file.name
                    
                    # Filter data for this date
                    day_data = df[df.index.date == date]
                    self.metadata_cache[date_str] = day_data
                
                print(f"  ✓ Loaded {csv_file.name}: {len(df)} records")
            
            except Exception as e:
                print(f"  ✗ Error loading {csv_file.name}: {e}")
                continue
        
        if not self.metadata_cache:
            raise ValueError("No valid metadata files could be loaded")
        
        print(f"Successfully loaded metadata for {len(self.metadata_cache)} day(s)")
    
    def _find_timestamp_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find the timestamp column in the dataframe."""
        # Common timestamp column names
        possible_names = [
            'timestamp', 'Timestamp', 'TIMESTAMP',
            'datetime', 'DateTime', 'date_time',
            'time', 'Time', 'DATE_TIME'
        ]
        
        for col in df.columns:
            if col in possible_names:
                return col
            # Check if column name contains 'time' or 'date'
            if 'time' in col.lower() or 'date' in col.lower():
                return col
        
        # If no obvious timestamp column, try the first column
        if len(df.columns) > 0:
            return df.columns[0]
        
        return None
    
    def parse_image_timestamp(self, image_path: Union[str, Path]) -> datetime:
        """
        Extract timestamp from thermal image filename.
        
        Expected format: thermal_YYYYMMDD_HHMMSS_frameXXX.tiff
        or similar variations.
        
        Args:
            image_path: Path to thermal image
        
        Returns:
            datetime: Extracted timestamp
        
        Raises:
            ValueError: If timestamp cannot be extracted
        """
        filename = Path(image_path).stem
        
        # Try different parsing strategies
        try:
            # Format: thermal_YYYYMMDD_HHMMSS_frameXXX
            parts = filename.split('_')
            
            # Find date and time parts
            date_str = None
            time_str = None
            
            for part in parts:
                # Look for 8-digit date (YYYYMMDD)
                if len(part) == 8 and part.isdigit():
                    date_str = part
                # Look for 6-digit time (HHMMSS)
                elif len(part) == 6 and part.isdigit():
                    time_str = part
            
            if date_str and time_str:
                timestamp_str = f"{date_str}_{time_str}"
                return datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            
            # Try alternative format: YYYY-MM-DD_HH-MM-SS
            for part in parts:
                if '-' in part and len(part) >= 8:
                    try:
                        return datetime.strptime(part, "%Y-%m-%d")
                    except:
                        pass
            
            raise ValueError(f"Could not parse timestamp from filename: {filename}")
        
        except Exception as e:
            raise ValueError(f"Error parsing timestamp from {filename}: {e}")
    
    def match_single_image(self, image_path: Union[str, Path], 
                          image_timestamp: Optional[datetime] = None) -> Dict:
        """
        Match a single thermal image with environmental metadata.
        
        Args:
            image_path: Path to thermal image
            image_timestamp: Pre-parsed timestamp (optional, will parse from filename if not provided)
        
        Returns:
            dict: Matched environmental data with all sensor measurements
        """
        # Parse timestamp if not provided
        if image_timestamp is None:
            image_timestamp = self.parse_image_timestamp(image_path)
        
        # Get date string for lookup
        date_str = image_timestamp.strftime('%Y-%m-%d')
        
        # Check if we have metadata for this date
        if date_str not in self.metadata_cache:
            raise ValueError(f"No metadata available for date: {date_str}")
        
        # Get metadata for this date
        day_metadata = self.metadata_cache[date_str]
        
        # Find closest timestamp using specified method
        if self.interpolation_method == 'nearest':
            # Find nearest timestamp within tolerance
            time_diffs = np.abs(day_metadata.index - image_timestamp)
            min_diff_idx = time_diffs.argmin()
            min_diff = time_diffs.iloc[min_diff_idx]
            
            if min_diff > self.time_tolerance:
                raise ValueError(
                    f"No metadata within {self.time_tolerance.total_seconds()}s of image timestamp. "
                    f"Closest match is {min_diff.total_seconds():.1f}s away."
                )
            
            matched_row = day_metadata.iloc[min_diff_idx]
        
        elif self.interpolation_method == 'linear':
            # Linear interpolation between surrounding timestamps
            matched_row = day_metadata.reindex(
                day_metadata.index.union([image_timestamp])
            ).interpolate(method='time').loc[image_timestamp]
        
        elif self.interpolation_method == 'forward':
            # Use the next available timestamp
            future_data = day_metadata[day_metadata.index >= image_timestamp]
            if len(future_data) == 0:
                raise ValueError(f"No metadata after timestamp: {image_timestamp}")
            matched_row = future_data.iloc[0]
        
        elif self.interpolation_method == 'backward':
            # Use the previous available timestamp
            past_data = day_metadata[day_metadata.index <= image_timestamp]
            if len(past_data) == 0:
                raise ValueError(f"No metadata before timestamp: {image_timestamp}")
            matched_row = past_data.iloc[-1]
        
        else:
            raise ValueError(f"Unknown interpolation method: {self.interpolation_method}")
        
        # Convert to dictionary and add metadata
        result = matched_row.to_dict()
        result['image_path'] = str(image_path)
        result['image_timestamp'] = image_timestamp
        result['matched_timestamp'] = matched_row.name
        result['time_difference_seconds'] = (matched_row.name - image_timestamp).total_seconds()
        
        return result
    
    def match_directory(self, image_dir: Union[str, Path], 
                       pattern: str = "*.tiff",
                       recursive: bool = True,
                       output_csv: Optional[str] = None) -> pd.DataFrame:
        """
        Match all thermal images in a directory with metadata.
        
        Args:
            image_dir: Directory containing thermal images
            pattern: Glob pattern for image files (default: "*.tiff")
            recursive: Search recursively in subdirectories
            output_csv: Optional path to save results as CSV
        
        Returns:
            pd.DataFrame: Matched data for all images
        """
        image_dir = Path(image_dir)
        
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        
        # Find all images
        if recursive:
            image_files = list(image_dir.rglob(pattern))
        else:
            image_files = list(image_dir.glob(pattern))
        
        if not image_files:
            raise ValueError(f"No images found in {image_dir} matching pattern '{pattern}'")
        
        print(f"Found {len(image_files)} image(s) to match...")
        
        # Match each image
        results = []
        matched_count = 0
        failed_count = 0
        
        for img_file in sorted(image_files):
            try:
                matched_data = self.match_single_image(img_file)
                results.append(matched_data)
                matched_count += 1
                
                if matched_count % 100 == 0:
                    print(f"  Matched {matched_count}/{len(image_files)} images...")
            
            except Exception as e:
                print(f"  ✗ Failed to match {img_file.name}: {e}")
                failed_count += 1
                continue
        
        print(f"\nMatching complete:")
        print(f"  ✓ Successful: {matched_count}")
        print(f"  ✗ Failed: {failed_count}")
        
        if not results:
            raise ValueError("No images could be matched with metadata")
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Save to CSV if requested
        if output_csv:
            df.to_csv(output_csv, index=False)
            print(f"\nResults saved to: {output_csv}")
        
        return df
    
    def get_metadata_summary(self) -> pd.DataFrame:
        """Get summary statistics for loaded metadata."""
        summaries = []
        
        for date_str, data in self.metadata_cache.items():
            summary = {
                'date': date_str,
                'num_records': len(data),
                'start_time': data.index.min(),
                'end_time': data.index.max(),
                'num_columns': len(data.columns)
            }
            
            # Add statistics for key columns if they exist
            key_cols = ['poa_irradiance', 'ambient_temp', 'soiling_ratio']
            for col in key_cols:
                if col in data.columns:
                    summary[f'{col}_mean'] = data[col].mean()
                    summary[f'{col}_std'] = data[col].std()
            
            summaries.append(summary)
        
        return pd.DataFrame(summaries)


def main():
    """Command-line interface for metadata matching."""
    parser = argparse.ArgumentParser(
        description="Match thermal images with environmental metadata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Match all images in a directory
  python match_metadata.py \\
    --image_dir ./data/raw/2024-12-20/08h00_30deg/10cm \\
    --metadata_dir ./data/environmental_metadata \\
    --output matched_metadata.csv
  
  # Match with linear interpolation
  python match_metadata.py \\
    --image_dir ./data/preprocessed \\
    --metadata_dir ./data/environmental_metadata \\
    --interpolation linear \\
    --output matched_data.csv
  
  # Match single image
  python match_metadata.py \\
    --image_path ./data/raw/.../thermal_20241220_080023_frame001.tiff \\
    --metadata_dir ./data/environmental_metadata

Output columns:
  - image_path: Path to thermal image
  - image_timestamp: Timestamp parsed from image filename
  - matched_timestamp: Timestamp from environmental data
  - time_difference_seconds: Time difference between image and data
  - poa_irradiance: Plane-of-Array Irradiance (W/m²)
  - ambient_temp: Ambient Temperature (°C)
  - soiling_ratio: Soiling Ratio (%)
  - [additional environmental measurements...]
        """
    )
    
    # Input arguments
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--image_dir',
        type=str,
        help='Directory containing thermal images'
    )
    input_group.add_argument(
        '--image_path',
        type=str,
        help='Single thermal image to match'
    )
    
    parser.add_argument(
        '--metadata_dir',
        type=str,
        required=True,
        help='Directory containing environmental metadata CSV files'
    )
    
    # Matching parameters
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.tiff',
        help='Glob pattern for image files (default: *.tiff)'
    )
    
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Search recursively in subdirectories'
    )
    
    parser.add_argument(
        '--tolerance',
        type=int,
        default=60,
        help='Time tolerance for matching in seconds (default: 60)'
    )
    
    parser.add_argument(
        '--interpolation',
        type=str,
        choices=['nearest', 'linear', 'forward', 'backward'],
        default='nearest',
        help='Temporal interpolation method (default: nearest)'
    )
    
    # Output arguments
    parser.add_argument(
        '--output',
        type=str,
        help='Output CSV file for matched data'
    )
    
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Print metadata summary statistics'
    )
    
    args = parser.parse_args()
    
    # Initialize matcher
    try:
        matcher = MetadataMatcher(
            metadata_dir=args.metadata_dir,
            time_tolerance_seconds=args.tolerance,
            interpolation_method=args.interpolation
        )
    except Exception as e:
        print(f"Error initializing matcher: {e}")
        sys.exit(1)
    
    # Print summary if requested
    if args.summary:
        print("\n" + "="*70)
        print("Metadata Summary")
        print("="*70)
        summary_df = matcher.get_metadata_summary()
        print(summary_df.to_string(index=False))
        print()
    
    # Match images
    try:
        if args.image_dir:
            # Match directory
            df = matcher.match_directory(
                image_dir=args.image_dir,
                pattern=args.pattern,
                recursive=args.recursive,
                output_csv=args.output
            )
            
            # Display sample results
            print("\n" + "="*70)
            print("Sample Matched Results (first 5 rows)")
            print("="*70)
            display_cols = ['image_path', 'image_timestamp', 'time_difference_seconds']
            # Add environmental columns if they exist
            for col in ['poa_irradiance', 'ambient_temp', 'soiling_ratio']:
                if col in df.columns:
                    display_cols.append(col)
            
            print(df[display_cols].head().to_string(index=False))
        
        else:
            # Match single image
            result = matcher.match_single_image(args.image_path)
            
            print("\n" + "="*70)
            print("Matched Metadata")
            print("="*70)
            for key, value in result.items():
                print(f"{key:30s}: {value}")
            
            # Save single result if output specified
            if args.output:
                df = pd.DataFrame([result])
                df.to_csv(args.output, index=False)
                print(f"\nResult saved to: {args.output}")
    
    except Exception as e:
        print(f"Error during matching: {e}")
        sys.exit(1)
    
    print("\n✓ Metadata matching complete!")


if __name__ == "__main__":
    main()