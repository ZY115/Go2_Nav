# ============================================================================
# Copyright (c) 2025-2026 Yihao Cai - All rights reserved
# SPDX-License-Identifier: Apache-2.0
# ============================================================================

"""
TensorBoard Log Reader and Analyzer
Used for reading and analyzing training logs from TensorBoard event files

Features:
- Load experiments by time-based index or name pattern
- Extract scalar metrics from TensorBoard logs
- Compute statistics and moving averages
- Export each metric to separate files (CSV/TXT) in organized folders
- Batch export all experiments
- Compare metrics across experiments
"""

import os
import glob
import numpy as np
from datetime import datetime
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


class LogReader:
    """
    A class for reading and analyzing TensorBoard logs
    
    This class provides methods to:
    - Load TensorBoard event files from experiment folders (sorted by time)
    - Extract scalar metrics
    - Compute statistics
    - Export data for further analysis
    """
    
    def __init__(self, log_dir, experiment_index=-1, experiment_name=None):
        """
        Initialize the log reader
        
        Args:
            log_dir (str): Path to the directory containing experiment folders
            experiment_index (int): Index of experiment to load (default: -1, most recent)
                                   Supports negative indexing:
                                   - 0: oldest experiment
                                   - -1: newest experiment (default)
                                   - 1: second oldest
                                   - -2: second newest
            experiment_name (str, optional): Name pattern to match experiment folder
                                            If provided, overrides experiment_index
        
        Examples:
            # Load most recent experiment
            reader = LogReader('../logs')
            
            # Load oldest experiment
            reader = LogReader('../logs', 0)
            
            # Load by name pattern
            reader = LogReader('../logs', experiment_name='dual')
            reader = LogReader('../logs', experiment_name='20251217')
        
        Raises:
            FileNotFoundError: If log_dir doesn't exist or contains no event files
            IndexError: If experiment_index is out of range
            ValueError: If experiment_name doesn't match any experiments
        """
        print("Initializing TensorBoard log reader...")
        self.base_log_dir = os.path.abspath(log_dir)
        
        # Find all experiment folders sorted by time
        self.experiment_dirs = self._find_and_sort_experiments()
        
        if not self.experiment_dirs:
            raise FileNotFoundError(
                f"No experiment folders with tfevents files found in {self.base_log_dir}"
            )
        
        print(f"Found {len(self.experiment_dirs)} experiment folder(s)")
        
        # Select experiment by name or index
        if experiment_name is not None:
            self.experiment_dir = self._select_by_name(experiment_name)
            self.experiment_index = self.experiment_dirs.index(self.experiment_dir)
        else:
            # Select experiment by index
            try:
                self.experiment_dir = self.experiment_dirs[experiment_index]
                self.experiment_index = experiment_index
            except IndexError:
                raise IndexError(
                    f"Experiment index {experiment_index} out of range. "
                    f"Valid range: 0 to {len(self.experiment_dirs)-1} "
                    f"or -{len(self.experiment_dirs)} to -1"
                )
        
        self.experiment_name = os.path.basename(self.experiment_dir)
        
        # Calculate actual index if negative
        actual_index = self.experiment_index if self.experiment_index >= 0 else len(self.experiment_dirs) + self.experiment_index
        print(f"Selected experiment [{actual_index}]: {self.experiment_name}")
        
        # Load event file
        self.event_acc = None
        self.event_file_path = None
        self.load_event_file()
    
    def _find_and_sort_experiments(self):
        """
        Find all experiment directories and sort by modification time
        
        Returns:
            list: List of experiment directory paths, sorted from oldest to newest
        """
        if not os.path.exists(self.base_log_dir):
            raise FileNotFoundError(f"Log directory does not exist: {self.base_log_dir}")
        
        # Find all subdirectories in log_dir
        subdirs = [
            os.path.join(self.base_log_dir, d)
            for d in os.listdir(self.base_log_dir)
            if os.path.isdir(os.path.join(self.base_log_dir, d))
        ]
        
        # Filter: only keep directories that contain tfevents files
        experiment_dirs = []
        for subdir in subdirs:
            try:
                has_events = any('tfevents' in f for f in os.listdir(subdir))
                if has_events:
                    experiment_dirs.append(subdir)
            except PermissionError:
                continue
        
        # Sort by modification time (oldest to newest)
        experiment_dirs.sort(key=lambda d: os.path.getmtime(d))
        
        return experiment_dirs
    
    def _select_by_name(self, name_pattern):
        """
        Select experiment by name pattern
        
        Args:
            name_pattern (str): Pattern to match in experiment folder name
        
        Returns:
            str: Selected experiment directory path
        
        Raises:
            ValueError: If no experiments match the pattern
        """
        matched = [
            exp for exp in self.experiment_dirs 
            if name_pattern.lower() in os.path.basename(exp).lower()
        ]
        
        if not matched:
            available = [os.path.basename(d) for d in self.experiment_dirs]
            raise ValueError(
                f"No experiments matching '{name_pattern}' found.\n"
                f"Available experiments: {available}"
            )
        
        if len(matched) > 1:
            print(f"Found {len(matched)} experiments matching '{name_pattern}':")
            for exp in matched:
                print(f"  - {os.path.basename(exp)}")
            print(f"Using most recent: {os.path.basename(matched[-1])}")
        
        # Return most recent match
        return matched[-1]
    
    def load_event_file(self):
        """
        Load the most recent TensorBoard event file from selected experiment
        
        Raises:
            FileNotFoundError: If no tfevents files are found
        """
        event_files = [
            os.path.join(self.experiment_dir, f)
            for f in os.listdir(self.experiment_dir)
            if 'tfevents' in f
        ]
        
        if not event_files:
            raise FileNotFoundError(
                f"No tfevents files found in {self.experiment_dir}"
            )
        
        # If multiple event files, select the most recent one
        self.event_file_path = sorted(event_files, key=os.path.getmtime)[-1]
        
        # Get modification time
        mtime = os.path.getmtime(self.experiment_dir)
        time_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"Experiment time: {time_str}")
        print(f"Loading: {os.path.basename(self.event_file_path)}")
        
        # Load event data
        self.event_acc = EventAccumulator(self.event_file_path)
        self.event_acc.Reload()
        
        tags = self.get_all_tags()
        print(f"Successfully loaded {len(tags)} metric(s): {tags}\n")
    
    def get_all_tags(self):
        """
        Get all available scalar tags
        
        Returns:
            list: List of tag names (e.g., ['train/loss', 'train/reward'])
        """
        if self.event_acc is None:
            return []
        return self.event_acc.Tags()['scalars']
    
    def get_scalar(self, tag):
        """
        Get scalar data for a specific tag
        
        Args:
            tag (str): Tag name (e.g., 'train/loss')
        
        Returns:
            list: List of ScalarEvent objects, each containing (wall_time, step, value)
        
        Raises:
            ValueError: If tag doesn't exist
        """
        if tag not in self.get_all_tags():
            raise ValueError(
                f"Tag '{tag}' not found. Available tags: {self.get_all_tags()}"
            )
        
        return self.event_acc.Scalars(tag)
    
    def get_scalar_values(self, tag):
        """
        Get scalar values as numpy arrays
        
        Args:
            tag (str): Tag name
        
        Returns:
            tuple: (steps, values) as numpy arrays
                - steps: Array of step numbers
                - values: Array of corresponding values
        """
        scalars = self.get_scalar(tag)
        steps = np.array([s.step for s in scalars])
        values = np.array([s.value for s in scalars])
        return steps, values
    
    def get_latest_value(self, tag):
        """
        Get the most recent value for a specific tag
        
        Args:
            tag (str): Tag name
        
        Returns:
            float: Latest value, or None if no data exists
        """
        scalars = self.get_scalar(tag)
        return float(scalars[-1].value) if scalars else None
    
    def get_statistics(self, tag):
        """
        Compute statistics for a specific tag
        
        Args:
            tag (str): Tag name
        
        Returns:
            dict: Dictionary containing:
                - min: Minimum value
                - max: Maximum value
                - mean: Average value
                - std: Standard deviation
                - latest: Most recent value
                - count: Number of data points
        """
        _, values = self.get_scalar_values(tag)
        
        return {
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'latest': float(values[-1]),
            'count': len(values)
        }
    
    def get_moving_average(self, tag, window=10):
        """
        Compute moving average for smoothing
        
        Args:
            tag (str): Tag name
            window (int): Window size for moving average
        
        Returns:
            tuple: (steps, smoothed_values)
                - steps: Array of step numbers (truncated by window size)
                - smoothed_values: Array of smoothed values
        """
        steps, values = self.get_scalar_values(tag)
        
        if len(values) < window:
            return steps, values
        
        # Compute moving average using convolution
        smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
        smoothed_steps = steps[window-1:]
        
        return smoothed_steps, smoothed
    
    def compare_tags(self, tags):
        """
        Compare latest values across multiple tags
        
        Args:
            tags (list): List of tag names to compare
        
        Returns:
            dict: Dictionary mapping tag names to their latest values
                  Returns None for tags that don't exist
        """
        result = {}
        for tag in tags:
            try:
                result[tag] = self.get_latest_value(tag)
            except ValueError as e:
                print(f"Warning: {e}")
                result[tag] = None
        
        return result
    
    def export_to_dict(self, tags=None):
        """
        Export data to dictionary format
        
        Args:
            tags (list, optional): List of tags to export. 
                                   If None, exports all available tags
        
        Returns:
            dict: Dictionary with structure:
                  {tag: {'steps': [...], 'values': [...]}}
        """
        if tags is None:
            tags = self.get_all_tags()
        
        data = {}
        for tag in tags:
            try:
                steps, values = self.get_scalar_values(tag)
                data[tag] = {
                    'steps': steps.tolist(),
                    'values': values.tolist()
                }
            except ValueError as e:
                print(f"Warning: {e}")
        
        return data
    
    def export_to_separate_files(
        self,
        output_dir=None,
        file_format='csv',
        tags=None,
        include_header=True,
        include_step=False,
    ):
        """
        Export each metric to a separate file in a folder named after the experiment
        
        Args:
            output_dir (str, optional): Base output directory. 
                                       If None, creates 'exported_data' in current directory
            file_format (str): File format, either 'csv' or 'txt' (default: 'csv')
            tags (list, optional): List of tags to export. If None, exports all available tags
            include_header (bool): Whether to write a header row
            include_step (bool): Whether to include the step column in exported rows
        
        Returns:
            str: Path to the created experiment folder
        
        Example:
            reader = LogReader('../logs', -1)
            reader.export_to_separate_files()
            
            # Output structure:
            # exported_data/
            # └── drl_student_dual_20251217_134450/
            #     ├── train_loss.csv
            #     ├── train_reward.csv
            #     ├── TrackingError_std.csv
            #     └── ...
        """
        if file_format not in ['csv', 'txt']:
            raise ValueError("file_format must be either 'csv' or 'txt'")
        
        # Set default output directory
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), 'exported_data')
        
        # Create experiment folder named after the experiment
        experiment_folder = os.path.join(output_dir, self.experiment_name)
        os.makedirs(experiment_folder, exist_ok=True)
        
        # Get tags to export
        if tags is None:
            tags = self.get_all_tags()
        
        print(f"\nExporting {len(tags)} metric(s) to: {experiment_folder}")
        print(f"{'='*70}")
        
        exported_files = []
        
        for tag in tags:
            try:
                steps, values = self.get_scalar_values(tag)
                
                # Convert tag name to filename
                # e.g., 'train/loss' -> 'train_loss.csv'
                # e.g., 'TrackingError/std' -> 'TrackingError_std.csv'
                filename = tag.replace('/', '_').replace('\\', '_')
                filepath = os.path.join(experiment_folder, f"{filename}.{file_format}")
                
                # Write to file
                with open(filepath, 'w') as f:
                    if file_format == 'csv':
                        # CSV format with optional header and step column
                        if include_header:
                            f.write("step,value\n" if include_step else "field\n")
                        for step, value in zip(steps, values):
                            if include_step:
                                f.write(f"{step},{value}\n")
                            else:
                                f.write(f"{value}\n")
                    else:  # txt format
                        # Space-separated format with optional header and step column
                        if include_header:
                            f.write("# step value\n" if include_step else "# field\n")
                        for step, value in zip(steps, values):
                            if include_step:
                                f.write(f"{step} {value}\n")
                            else:
                                f.write(f"{value}\n")
                
                exported_files.append(filepath)
                print(f"  ✓ {filename}.{file_format} ({len(values)} data points)")
                
            except ValueError as e:
                print(f"  ✗ {tag}: {e}")
        
        print(f"{'='*70}")
        print(f"Successfully exported {len(exported_files)} file(s)")
        print(f"Location: {experiment_folder}\n")
        
        return experiment_folder
    
    def export_all_to_single_csv(self, output_dir=None, tags=None, include_header=False, include_step=False):
        """
        Export all metrics to a single CSV file (wide format)
        
        Args:
            output_dir (str, optional): Output directory. 
                                       If None, creates 'exported_data' in current directory
            tags (list, optional): List of tags to export. If None, exports all available tags
            include_header (bool): Whether to write a header row
            include_step (bool): Whether to include the step column in exported rows
        
        Returns:
            str: Path to the created CSV file
        
        Example output CSV structure:
            step,train_loss,train_reward,TrackingError_std
            0,1.234,5.678,0.123
            1,1.123,5.789,0.115
            ...
        """
        # Set default output directory
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), 'exported_data')
        
        # Create experiment folder
        experiment_folder = os.path.join(output_dir, self.experiment_name)
        os.makedirs(experiment_folder, exist_ok=True)
        
        # Get tags to export
        if tags is None:
            tags = self.get_all_tags()
        
        # Collect all data
        data_dict = {}
        all_steps = set()
        
        for tag in tags:
            try:
                steps, values = self.get_scalar_values(tag)
                tag_name = tag.replace('/', '_').replace('\\', '_')
                data_dict[tag_name] = dict(zip(steps, values))
                all_steps.update(steps)
            except ValueError as e:
                print(f"Warning: {e}")
        
        # Sort steps
        all_steps = sorted(all_steps)
        
        # Create CSV file
        filepath = os.path.join(experiment_folder, 'all_metrics.csv')
        
        with open(filepath, 'w') as f:
            # Write header
            if include_header:
                header_fields = list(data_dict.keys())
                if include_step:
                    header_fields = ['step'] + header_fields
                f.write(','.join(header_fields) + '\n')
            
            # Write data
            for step in all_steps:
                row = [str(step)] if include_step else []
                for tag_name in data_dict.keys():
                    value = data_dict[tag_name].get(step, '')
                    row.append(str(value) if value != '' else '')
                f.write(','.join(row) + '\n')
        
        print(f"\nExported all metrics to single CSV:")
        print(f"  {filepath}")
        print(f"  Metrics: {len(data_dict)}")
        print(f"  Steps: {len(all_steps)}\n")
        
        return filepath
    
    def export_summary_stats(self, output_dir=None, tags=None, include_header=False):
        """
        Export summary statistics for all metrics to a CSV file
        
        Args:
            output_dir (str, optional): Output directory
            tags (list, optional): List of tags to export. If None, exports all available tags
            include_header (bool): Whether to write a header row
        
        Returns:
            str: Path to the created CSV file
        
        Example output:
            metric,min,max,mean,std,latest,count
            train_loss,0.123,1.234,0.567,0.089,0.234,1000
            train_reward,5.678,20.123,12.345,3.456,15.678,1000
        """
        # Set default output directory
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), 'exported_data')
        
        # Create experiment folder
        experiment_folder = os.path.join(output_dir, self.experiment_name)
        os.makedirs(experiment_folder, exist_ok=True)
        
        # Get tags to export
        if tags is None:
            tags = self.get_all_tags()
        
        # Create CSV file
        filepath = os.path.join(experiment_folder, 'summary_statistics.csv')
        
        with open(filepath, 'w') as f:
            # Write header
            if include_header:
                f.write("metric,min,max,mean,std,latest,count\n")
            
            # Write statistics for each metric
            for tag in tags:
                try:
                    stats = self.get_statistics(tag)
                    tag_name = tag.replace('/', '_').replace('\\', '_')
                    f.write(
                        f"{tag_name},"
                        f"{stats['min']:.6f},"
                        f"{stats['max']:.6f},"
                        f"{stats['mean']:.6f},"
                        f"{stats['std']:.6f},"
                        f"{stats['latest']:.6f},"
                        f"{stats['count']}\n"
                    )
                except ValueError as e:
                    print(f"Warning: {e}")
        
        print(f"\nExported summary statistics:")
        print(f"  {filepath}")
        print(f"  Metrics: {len(tags)}\n")
        
        return filepath
    
    def export_all(
        self,
        output_dir=None,
        file_format='csv',
        tags=None,
        include_header=False,
        include_step=False,
    ):
        """
        Export all data in multiple formats:
        1. Each metric in separate files
        2. All metrics in single CSV (wide format)
        3. Summary statistics
        
        Args:
            output_dir (str, optional): Output directory
            file_format (str): File format for individual files ('csv' or 'txt')
            tags (list, optional): List of tags to export. If None, exports all available tags
            include_header (bool): Whether to write header rows in exported files
            include_step (bool): Whether to include the step column in exported rows
        
        Returns:
            dict: Dictionary with paths to all created files
        
        Example:
            reader = LogReader('../logs', -1)
            paths = reader.export_all()
            
            # Output structure:
            # exported_data/
            # └── drl_student_dual_20251217_134450/
            #     ├── train_loss.csv
            #     ├── train_reward.csv
            #     ├── TrackingError_std.csv
            #     ├── all_metrics.csv
            #     └── summary_statistics.csv
        """
        print(f"\n{'='*70}")
        print(f"Exporting all data for experiment: {self.experiment_name}")
        print(f"{'='*70}")
        
        # Export individual files
        experiment_folder = self.export_to_separate_files(
            output_dir=output_dir,
            file_format=file_format,
            tags=tags,
            include_header=include_header,
            include_step=include_step,
        )
        
        # Export combined CSV
        combined_csv = self.export_all_to_single_csv(
            output_dir=output_dir,
            tags=tags,
            include_header=include_header,
            include_step=include_step,
        )
        
        # Export summary statistics
        summary_csv = self.export_summary_stats(
            output_dir=output_dir,
            tags=tags,
            include_header=include_header,
        )
        
        print(f"{'='*70}")
        print(f"All exports completed!")
        print(f"{'='*70}\n")
        
        return {
            'experiment_folder': experiment_folder,
            'combined_csv': combined_csv,
            'summary_csv': summary_csv
        }
    
    def export_to_dataframe(self, tags=None):
        """
        Export data to pandas DataFrame (long format)
        
        Args:
            tags (list, optional): List of tags to export.
                                   If None, exports all available tags
        
        Returns:
            pd.DataFrame: DataFrame with columns ['tag', 'step', 'value']
        
        Raises:
            ImportError: If pandas is not installed
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for this function. "
                "Install it with: pip install pandas"
            )
        
        if tags is None:
            tags = self.get_all_tags()
        
        # Collect all data
        all_data = []
        for tag in tags:
            try:
                steps, values = self.get_scalar_values(tag)
                for step, value in zip(steps, values):
                    all_data.append({
                        'tag': tag,
                        'step': step,
                        'value': value
                    })
            except ValueError as e:
                print(f"Warning: {e}")
        
        return pd.DataFrame(all_data)
    
    def print_summary(self):
        """
        Print a summary of all available metrics
        
        Displays:
        - Experiment name and time
        - Log file path
        - Number of available tags
        - Statistics for each tag (latest, mean, std, range, count)
        """
        print(f"\n{'='*70}")
        print(f"Experiment: {self.experiment_name}")
        print(f"Log file: {os.path.basename(self.event_file_path)}")
        mtime = os.path.getmtime(self.experiment_dir)
        time_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        print(f"Time: {time_str}")
        print(f"{'='*70}")
        
        tags = self.get_all_tags()
        
        if not tags:
            print("\nNo metrics found in this experiment.")
            return
        
        print(f"\nFound {len(tags)} metric(s):\n")
        
        for tag in tags:
            stats = self.get_statistics(tag)
            print(f"  {tag}:")
            print(f"    Latest:  {stats['latest']:.4f}")
            print(f"    Mean:    {stats['mean']:.4f} ± {stats['std']:.4f}")
            print(f"    Range:   [{stats['min']:.4f}, {stats['max']:.4f}]")
            print(f"    Count:   {stats['count']}")
            print()
    
    def list_all_experiments(self):
        """
        List all available experiments with details
        
        Displays:
        - Index (both positive and negative)
        - Experiment name
        - Modification time
        - Current experiment marker
        """
        print(f"\n{'='*70}")
        print(f"All experiments in {os.path.basename(self.base_log_dir)}:")
        print(f"{'='*70}\n")
        
        for i, exp_dir in enumerate(self.experiment_dirs):
            exp_name = os.path.basename(exp_dir)
            mtime = os.path.getmtime(exp_dir)
            time_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
            
            # Check if this is the current experiment
            is_current = (exp_dir == self.experiment_dir)
            marker = " <-- CURRENT" if is_current else ""
            
            # Also show negative index
            neg_index = i - len(self.experiment_dirs)
            
            print(f"[{i:2d}] (or [{neg_index:2d}]) {exp_name}{marker}")
            print(f"     Time: {time_str}")
        
        print(f"\n{'='*70}")
        print("Usage examples:")
        print("  reader = LogReader('../logs', 0)   # Load oldest")
        print("  reader = LogReader('../logs', -1)  # Load newest (default)")
        print("  reader = LogReader('../logs', -2)  # Load second newest")
        print("  reader = LogReader('../logs', experiment_name='dual')  # Load by name")
        print(f"{'='*70}\n")
    
    def __repr__(self):
        """String representation of the LogReader"""
        return (
            f"LogReader(experiment='{self.experiment_name}', "
            f"index={self.experiment_index}, tags={len(self.get_all_tags())})"
        )


# Standalone helper functions
def list_experiments(log_dir):
    """
    List all experiments in a directory sorted by time
    
    Args:
        log_dir (str): Base log directory
    
    Returns:
        list: List of experiment directory paths (sorted by time)
    """
    log_dir = os.path.abspath(log_dir)
    
    if not os.path.exists(log_dir):
        print(f"Directory does not exist: {log_dir}")
        return []
    
    # Find all subdirectories
    subdirs = [
        os.path.join(log_dir, d)
        for d in os.listdir(log_dir)
        if os.path.isdir(os.path.join(log_dir, d))
    ]
    
    # Filter: only keep directories with tfevents files
    experiment_dirs = []
    for subdir in subdirs:
        try:
            has_events = any('tfevents' in f for f in os.listdir(subdir))
            if has_events:
                experiment_dirs.append(subdir)
        except PermissionError:
            continue
    
    # Sort by modification time (oldest to newest)
    experiment_dirs.sort(key=lambda d: os.path.getmtime(d))
    
    if not experiment_dirs:
        print(f"No experiments found in {log_dir}")
        return []
    
    print(f"\n{'='*70}")
    print(f"Found {len(experiment_dirs)} experiment(s) in {os.path.basename(log_dir)}:")
    print(f"{'='*70}\n")
    
    for i, exp_dir in enumerate(experiment_dirs):
        exp_name = os.path.basename(exp_dir)
        mtime = os.path.getmtime(exp_dir)
        time_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        
        # Show both positive and negative index
        neg_index = i - len(experiment_dirs)
        
        print(f"[{i:2d}] (or [{neg_index:2d}]) {exp_name}")
        print(f"     Time: {time_str}")
        print()
    
    return experiment_dirs


def export_all_experiments(
    log_dir,
    output_dir=None,
    file_format='csv',
    tags=None,
    name_pattern=None,
    include_header=False,
    include_step=False,
):
    """
    Export all experiments (or filtered by name pattern) to separate folders
    
    Args:
        log_dir (str): Base log directory
        output_dir (str, optional): Output directory. If None, creates 'exported_data'
        file_format (str): File format for individual files ('csv' or 'txt')
        tags (list, optional): List of tags to export. If None, exports all available tags
        name_pattern (str, optional): Only export experiments matching this pattern
        include_header (bool): Whether to write header rows in exported files
        include_step (bool): Whether to include the step column in exported rows
    
    Returns:
        dict: Dictionary mapping experiment names to their export paths
    
    Example:
        # Export all experiments
        export_all_experiments('../logs')
        
        # Export only 'dual' experiments
        export_all_experiments('../logs', name_pattern='dual')
        
        # Export to custom directory
        export_all_experiments('../logs', output_dir='/home/user/exports')
        
        # Output structure:
        # exported_data/
        # ├── drl_student_cognitive_20251216_184120/
        # │   ├── train_loss.csv
        # │   ├── train_reward.csv
        # │   └── ...
        # ├── drl_student_dual_20251216_190132/
        # │   ├── train_loss.csv
        # │   └── ...
        # └── drl_student_dual_20251217_134450/
        #     ├── train_loss.csv
        #     └── ...
    """
    log_dir = os.path.abspath(log_dir)
    
    if not os.path.exists(log_dir):
        print(f"Directory does not exist: {log_dir}")
        return {}
    
    # Find all subdirectories
    subdirs = [
        os.path.join(log_dir, d)
        for d in os.listdir(log_dir)
        if os.path.isdir(os.path.join(log_dir, d))
    ]
    
    # Filter: only keep directories with tfevents files
    experiment_dirs = []
    for subdir in subdirs:
        try:
            has_events = any('tfevents' in f for f in os.listdir(subdir))
            if has_events:
                # Apply name pattern filter if provided
                if name_pattern is None or name_pattern.lower() in os.path.basename(subdir).lower():
                    experiment_dirs.append(subdir)
        except PermissionError:
            continue
    
    # Sort by modification time
    experiment_dirs.sort(key=lambda d: os.path.getmtime(d))
    
    if not experiment_dirs:
        print(f"No experiments found in {log_dir}")
        if name_pattern:
            print(f"(with pattern: '{name_pattern}')")
        return {}
    
    print(f"\n{'='*70}")
    print(f"Batch Export: {len(experiment_dirs)} experiment(s)")
    if name_pattern:
        print(f"Filter pattern: '{name_pattern}'")
    print(f"{'='*70}\n")
    
    results = {}
    failed = []
    
    for i, exp_dir in enumerate(experiment_dirs, 1):
        exp_name = os.path.basename(exp_dir)
        print(f"[{i}/{len(experiment_dirs)}] Processing: {exp_name}")
        
        try:
            # Create a temporary reader for this experiment
            reader = LogReader(log_dir, experiment_index=experiment_dirs.index(exp_dir))
            
            # Export all data
            paths = reader.export_all(
                output_dir=output_dir,
                file_format=file_format,
                tags=tags,
                include_header=include_header,
                include_step=include_step,
            )
            results[exp_name] = paths
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            failed.append(exp_name)
            continue
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"Batch Export Summary")
    print(f"{'='*70}")
    print(f"Total experiments: {len(experiment_dirs)}")
    print(f"Successfully exported: {len(results)}")
    print(f"Failed: {len(failed)}")
    
    if failed:
        print(f"\nFailed experiments:")
        for name in failed:
            print(f"  - {name}")
    
    if output_dir:
        print(f"\nOutput directory: {output_dir}")
    else:
        print(f"\nOutput directory: {os.path.join(os.getcwd(), 'exported_data')}")
    
    print(f"{'='*70}\n")
    
    return results


def compare_experiments(log_dir, indices, tag):
    """
    Compare a specific metric across multiple experiments
    
    Args:
        log_dir (str): Base log directory
        indices (list): List of experiment indices to compare
        tag (str): Tag name to compare
    
    Returns:
        dict: Dictionary mapping experiment names to their latest values
    
    Example:
        compare_experiments('../logs', [0, -1], 'train/loss')
    """
    results = {}
    
    for idx in indices:
        try:
            reader = LogReader(log_dir, idx)
            exp_name = os.path.basename(reader.experiment_dir)
            value = reader.get_latest_value(tag)
            results[exp_name] = value
        except Exception as e:
            print(f"Error loading experiment {idx}: {e}")
    
    return results


def plot_comparison(log_dir, indices, tag, window=None):
    """
    Plot comparison of a metric across multiple experiments
    
    Args:
        log_dir (str): Base log directory
        indices (list): List of experiment indices to compare
        tag (str): Tag name to plot
        window (int, optional): Window size for moving average smoothing
    
    Raises:
        ImportError: If matplotlib is not installed
    
    Example:
        plot_comparison('../logs', [0, -2, -1], 'train/loss', window=10)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for this function. "
            "Install it with: pip install matplotlib"
        )
    
    plt.figure(figsize=(12, 6))
    
    for idx in indices:
        try:
            reader = LogReader(log_dir, idx)
            exp_name = os.path.basename(reader.experiment_dir)
            
            if window:
                steps, values = reader.get_moving_average(tag, window)
                label = f"{exp_name} (smoothed)"
            else:
                steps, values = reader.get_scalar_values(tag)
                label = exp_name
            
            plt.plot(steps, values, label=label, alpha=0.7)
            
        except Exception as e:
            print(f"Error loading experiment {idx}: {e}")
    
    plt.xlabel('Step')
    plt.ylabel(tag)
    plt.title(f'Comparison: {tag}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    log_dir = '/home/edgebot/edge_ws/data/end2end/cognitive_wo_blending/outputs/logs'
    
    # print("="*70)
    # print("Example 1: List all experiments")
    # print("="*70)
    # experiments = list_experiments(log_dir)
    
    # print("\n" + "="*70)
    # print("Example 2: Load experiment by name pattern")
    # print("="*70)
    # try:
    #     reader = LogReader(log_dir, experiment_name='dual')
    #     reader.print_summary()
    # except ValueError as e:
    #     print(f"Error: {e}")
    
    # print("\n" + "="*70)
    # print("Example 3: Load newest experiment and export")
    # print("="*70)
    # reader = LogReader(log_dir)  # Load most recent
    # reader.export_all()
    
    print("\n" + "="*70)
    print("Example 4: Export ALL experiments")
    print("="*70)
    export_all_experiments(log_dir, file_format='csv', output_dir='cognitive_wo_blending')
    
    # print("\n" + "="*70)
    # print("Example 5: Export only 'dual' experiments")
    # print("="*70)
    # export_all_experiments(log_dir, name_pattern='dual', file_format='csv')
    
    # print("\n" + "="*70)
    # print("Example 6: Export only 'cognitive' experiments to custom directory")
    # print("="*70)
    # export_all_experiments(
    #     log_dir, 
    #     output_dir='./my_exports',
    #     name_pattern='cognitive',
    #     file_format='csv'
    # )
    
    # print("\n" + "="*70)
    # print("Example 7: Export specific tags from all experiments")
    # print("="*70)
    # specific_tags = ['train/loss', 'train/reward']
    # export_all_experiments(log_dir, tags=specific_tags)
