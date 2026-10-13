# ============================================================================
# Copyright (c) 2025-2026 Yihao Cai - All rights reserved
# SPDX-License-Identifier: Apache-2.0
# ============================================================================

import os
import time
import statistics
import numpy as np
import pandas as pd
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


class Logger:
    """
    Lightweight RL logger inspired by rsl_rl OnPolicyRunner.log(),
    but simplified and adapted for off-policy training loops.

    Features:
    - TensorBoard logging
    - Pretty console printout
    - reward buffer, episode length buffer
    - loss metrics logging
    - performance statistics (FPS, iteration time)
    """

    def __init__(self, log_dir: str, flush_secs=10, device="cpu"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=log_dir, flush_secs=flush_secs)
        self.device = device

        # counters
        self.total_time = 0.0
        self.total_env_steps = 0
        self.current_iteration = 0

        # episode statistics
        self.rewbuffer = deque(maxlen=100)
        self.lenbuffer = deque(maxlen=100)

        print(f"[EdgeDLLogger] Logging to: {self.log_dir}")

    # ------------------------------------------------------------------ #
    # Logging API
    # ------------------------------------------------------------------ #

    def add_episode_info(self, reward, length):
        """Add completed episode stats."""
        self.rewbuffer.append(float(reward))
        self.lenbuffer.append(float(length))

    def log_losses(self, loss_dict: dict, step: int):
        """Log loss scalars."""
        for key, value in loss_dict.items():
            self.writer.add_scalar(f"Loss/{key}", value, step)

    def log_train_metrics(self, step: int):
        """Log reward & episode length."""
        if len(self.rewbuffer) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(self.rewbuffer), step)
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(self.lenbuffer), step)

    def log_policy_stats(self, noise_std: float, step: int):
        """Log policy noise/entropy/etc."""
        self.writer.add_scalar("Policy/noise_std", noise_std, step)

    def log_performance(self, fps: float, step: int, collection_time: float, learn_time: float):
        """Log speed and timing info."""
        self.writer.add_scalar("Perf/fps", fps, step)
        self.writer.add_scalar("Perf/collection_time", collection_time, step)
        self.writer.add_scalar("Perf/learn_time", learn_time, step)

    # ------------------------------------------------------------------ #
    # Pretty Print like rsl_rl
    # ------------------------------------------------------------------ #

    def print_summary(
        self,
        it: int,
        total_iterations: int,
        fps: float,
        loss_dict: dict,
        noise_std: float,
        collection_time: float,
        learn_time: float
    ):
        pad = 32
        width = 70
        s = f" Learning iteration {it}/{total_iterations} "

        msg = (
            f"\n{'#' * width}\n"
            f"{s.center(width)}\n\n"
            f"{'Computation:':>{pad}} {fps:.0f} steps/s "
            f"(collection {collection_time:.3f}s, learn {learn_time:.3f}s)\n"
            f"{'Policy noise std:':>{pad}} {noise_std:.3f}\n"
        )

        for key, val in loss_dict.items():
            msg += f"{('Loss ' + key + ':'):>{pad}} {val:.4f}\n"

        if len(self.rewbuffer) > 0:
            msg += f"{'Mean reward:':>{pad}} {statistics.mean(self.rewbuffer):.2f}\n"
            msg += f"{'Mean episode length:':>{pad}} {statistics.mean(self.lenbuffer):.2f}\n"

        msg += (
            f"{'-' * width}\n"
            f"{'Total env steps:':>{pad}} {self.total_env_steps}\n"
            f"{'Total time:':>{pad}} {time.strftime('%H:%M:%S', time.gmtime(self.total_time))}\n"
        )

        print(msg)

    # ------------------------------------------------------------------ #
    # Main entry called by training loop
    # ------------------------------------------------------------------ #

    def log_iteration(
        self,
        *,
        it: int,
        total_iterations: int,
        loss_dict: dict,
        noise_std: float,
        env_steps_this_iter: int,
        collection_time: float,
        learn_time: float
    ):
        """Main logging entry point (similar to rsl_rl runner.log)."""

        # update counters
        self.current_iteration = it
        self.total_env_steps += env_steps_this_iter
        iter_time = collection_time + learn_time
        self.total_time += iter_time
        fps = env_steps_this_iter / iter_time

        # Write to TF summary
        self.log_losses(loss_dict, it)
        self.log_train_metrics(it)
        self.log_policy_stats(noise_std, it)
        self.log_performance(fps, it, collection_time, learn_time)

        # Print to console
        self.print_summary(
            it=it,
            total_iterations=total_iterations,
            fps=fps,
            loss_dict=loss_dict,
            noise_std=noise_std,
            collection_time=collection_time,
            learn_time=learn_time,
        )

    def close(self):
        self.writer.close()

    # ------------------------------------------------------------------ #
    # TensorBoard Data Export Functions
    # ------------------------------------------------------------------ #
    def export_to_csv(self, output_dir=None, separate_files=True):
        """
        Export TensorBoard data to CSV files.
        
        Args:
            output_dir: Output directory (default: log_dir/exported_csv)
            separate_files: If True, create separate CSV for each metric;
                          If False, combine all metrics into one CSV
        """
        if output_dir is None:
            output_dir = os.path.join(self.log_dir, 'exported_csv')
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Flush writer to ensure all data is written
        self.writer.flush()
        
        try:
            # Load TensorBoard event file
            ea = EventAccumulator(self.log_dir)
            ea.Reload()
            
            tags = ea.Tags()['scalars']
            
            if not tags:
                print("[Logger] No scalar data found to export.")
                return
            
            print(f"[Logger] Found {len(tags)} metrics to export:")
            for tag in tags:
                print(f"  - {tag}")
            
            if separate_files:
                # Export each metric to separate CSV
                for tag in tags:
                    events = ea.Scalars(tag)
                    
                    data = {
                        'step': [e.step for e in events],
                        'value': [e.value for e in events],
                        'wall_time': [e.wall_time for e in events]
                    }
                    df = pd.DataFrame(data)
                    
                    # Create safe filename
                    safe_tag = tag.replace('/', '_').replace('\\', '_')
                    csv_path = os.path.join(output_dir, f'{safe_tag}.csv')
                    df.to_csv(csv_path, index=False)
                    print(f"[Logger] Exported: {csv_path}")
            else:
                # Combine all metrics into one CSV
                all_data = []
                for tag in tags:
                    events = ea.Scalars(tag)
                    for e in events:
                        all_data.append({
                            'metric': tag,
                            'step': e.step,
                            'value': e.value,
                            'wall_time': e.wall_time
                        })
                
                df = pd.DataFrame(all_data)
                csv_path = os.path.join(output_dir, 'all_metrics.csv')
                df.to_csv(csv_path, index=False)
                print(f"[Logger] Exported all metrics to: {csv_path}")
            
            print(f"[Logger] Export completed. Files saved to: {output_dir}")
            
        except Exception as e:
            print(f"[Logger] Error during export: {e}")

    def export_to_txt(self, output_dir=None):
        """
        Export TensorBoard data to TXT files.
        
        Args:
            output_dir: Output directory (default: log_dir/exported_txt)
        """
        if output_dir is None:
            output_dir = os.path.join(self.log_dir, 'exported_txt')
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Flush writer to ensure all data is written
        self.writer.flush()
        
        try:
            ea = EventAccumulator(self.log_dir)
            ea.Reload()
            
            tags = ea.Tags()['scalars']
            
            if not tags:
                print("[Logger] No scalar data found to export.")
                return
            
            for tag in tags:
                events = ea.Scalars(tag)
                
                safe_tag = tag.replace('/', '_').replace('\\', '_')
                txt_path = os.path.join(output_dir, f'{safe_tag}.txt')
                
                with open(txt_path, 'w') as f:
                    f.write(f"Metric: {tag}\n")
                    f.write("=" * 80 + "\n")
                    f.write(f"{'Step':<10} {'Value':<20} {'Wall Time':<20}\n")
                    f.write("-" * 80 + "\n")
                    
                    for e in events:
                        f.write(f"{e.step:<10} {e.value:<20.6f} {e.wall_time:<20.6f}\n")
                
                print(f"[Logger] Exported: {txt_path}")
            
            print(f"[Logger] Export completed. Files saved to: {output_dir}")
            
        except Exception as e:
            print(f"[Logger] Error during export: {e}")

    def export_summary_statistics(self, output_file=None):
        """
        Export summary statistics of all metrics to CSV.
        
        Args:
            output_file: Output CSV file path (default: log_dir/summary_statistics.csv)
        """
        if output_file is None:
            output_file = os.path.join(self.log_dir, 'summary_statistics.csv')
        
        # Flush writer to ensure all data is written
        self.writer.flush()
        
        try:
            ea = EventAccumulator(self.log_dir)
            ea.Reload()
            
            tags = ea.Tags()['scalars']
            
            if not tags:
                print("[Logger] No scalar data found to export.")
                return
            
            summary_data = []
            
            for tag in tags:
                events = ea.Scalars(tag)
                values = [e.value for e in events]
                steps = [e.step for e in events]
                
                if not values:
                    continue
                
                import numpy as np
                stats = {
                    'metric': tag,
                    'count': len(values),
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'final_value': values[-1],
                    'final_step': steps[-1]
                }
                summary_data.append(stats)
            
            df = pd.DataFrame(summary_data)
            df.to_csv(output_file, index=False)
            
            print(f"[Logger] Summary statistics exported to: {output_file}")
            print("\nSummary Statistics:")
            print(df.to_string(index=False))
            
        except Exception as e:
            print(f"[Logger] Error during export: {e}")

    def close(self):
        """Close the logger and optionally export data."""
        self.writer.close()
        print("[Logger] TensorBoard writer closed.")
    
    def close_and_export(self, export_csv=True, export_txt=False, export_summary=True):
        """
        Close the logger and export data.
        
        Args:
            export_csv: Whether to export CSV files
            export_txt: Whether to export TXT files
            export_summary: Whether to export summary statistics
        """
        self.writer.close()
        
        print("\n" + "=" * 70)
        print("Exporting TensorBoard data...".center(70))
        print("=" * 70)
        
        if export_csv:
            self.export_to_csv(separate_files=True)
        
        if export_txt:
            self.export_to_txt()
        
        if export_summary:
            self.export_summary_statistics()
        
        print("\n" + "=" * 70)
        print("Export completed!".center(70))
        print("=" * 70 + "\n")
        
        
        
if __name__ == "__main__":
    # Example 1: Normal training with logging
    print("=" * 80)
    print("Example 1: Training with logging")
    print("=" * 80)
    
    logger = Logger(log_dir="./runs/example_experiment")
    
    # Simulate training loop
    for it in range(5):
        # Simulate episode completions
        logger.add_episode_info(reward=100 + it * 10, length=200 + it * 5)
        
        # Log iteration
        logger.log_iteration(
            it=it,
            total_iterations=5,
            loss_dict={"actor": 0.1 - it * 0.01, "critic": 0.2 - it * 0.02},
            noise_std=0.5 - it * 0.05,
            env_steps_this_iter=1000,
            collection_time=0.5,
            learn_time=0.3
        )
    
    # Close and export
    logger.close_and_export(export_csv=True, export_txt=False, export_summary=True)
    
    print("\n" + "=" * 80)
    print("Example 2: Export existing logs")
    print("=" * 80)
    
    # Export existing logs without creating new Logger
    Logger.export_existing_logs('./runs/example_experiment', formats=['all'])
    
    print("\n" + "=" * 80)
    print("Example 3: Export from different directory using instance")
    print("=" * 80)
    
    # Create logger for new experiment but export old one
    logger2 = Logger(log_dir="./runs/new_experiment")
    logger2.export_all(log_dir='./runs/example_experiment')
    logger2.close()