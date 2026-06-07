"""
================================================================================
PyRate DES Tutorial - Notebook 2: Create Environmental Predictor Files
================================================================================

This notebook provides functions to:
1. Load time series data (paleoclimate, sea level, etc.)
2. Bin predictor values to match DES time bins
3. Create separate files per area (if needed)
4. Handle area-specific vs. shared predictors
5. Export in PyRateDES-compatible format

Author: PyRate community
Date: 2024
================================================================================
"""

import pandas as pd
import numpy as np
import os
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# PART 1: LOAD AND PREPARE PREDICTOR DATA
# ============================================================================

class PredictorProcessor:
    """
    Manage environmental predictor time series for DES analysis.
    """
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.predictors = {}  # dict: predictor_name → DataFrame
        self.bin_times = None
        self.bin_size = None
    
    def load_predictor(self, name, filepath, time_col='time', value_col='value'):
        """
        Load a time series predictor file.
        
        Parameters
        ----------
        name : str
            Predictor name (e.g., 'sea_level', 'temperature')
        filepath : str
            Path to data file (tab or comma-separated)
        time_col : str
            Column name for time (Ma before present)
        value_col : str
            Column name for predictor value
        
        Returns
        -------
        DataFrame
            Loaded predictor data
        
        Example
        -------
        >>> proc = PredictorProcessor()
        >>> proc.load_predictor('sea_level', 'sea_level.txt')
        >>> proc.load_predictor('temperature', 'temp.csv')
        """
        
        try:
            # Try tab-separated first
            df = pd.read_csv(filepath, sep='\t')
            if len(df.columns) == 1:
                # Try comma-separated
                df = pd.read_csv(filepath, sep=',')
        except:
            print(f"✗ Error loading {filepath}")
            return None
        
        # Check columns
        if time_col not in df.columns or value_col not in df.columns:
            print(f"✗ Missing columns. Expected '{time_col}' and '{value_col}'")
            print(f"  Available: {df.columns.tolist()}")
            return None
        
        # Extract and sort by time
        df_pred = df[[time_col, value_col]].copy()
        df_pred.columns = ['time', 'value']
        df_pred = df_pred.sort_values('time')
        
        # Remove duplicates and NaN
        df_pred = df_pred.drop_duplicates(subset=['time'])
        df_pred = df_pred.dropna()
        
        self.predictors[name] = df_pred
        
        if self.verbose:
            print(f"✓ Loaded '{name}' from {filepath}")
            print(f"  Time range: {df_pred['time'].min():.2f} - {df_pred['time'].max():.2f} Ma")
            print(f"  Value range: {df_pred['value'].min():.3f} - {df_pred['value'].max():.3f}")
        
        return df_pred
    
    def set_bin_times(self, bin_times):
        """
        Set the time bins for binning predictors.
        
        Parameters
        ----------
        bin_times : array-like
            Time bin boundaries (oldest to youngest, Ma before present)
        
        Example
        -------
        >>> proc.set_bin_times(np.arange(0, 50, 2.5))
        """
        
        self.bin_times = np.asarray(bin_times)
        self.bin_size = self.bin_times[1] - self.bin_times[0] if len(self.bin_times) > 1 else 1
        
        if self.verbose:
            print(f"✓ Set bin times: {len(self.bin_times)} bins of {self.bin_size:.2f} Myr")
    
    # ========================================================================
    # PART 2: BIN PREDICTOR VALUES
    # ========================================================================
    
    def bin_predictor(self, name, method='mean'):
        """
        Bin a predictor to match DES time bins.
        
        Parameters
        ----------
        name : str
            Predictor name
        method : str
            Binning method: 'mean', 'median', 'max', 'min', 'interpolate'
        
        Returns
        -------
        DataFrame
            Binned values (time, value)
        
        Example
        -------
        >>> binned_sl = proc.bin_predictor('sea_level', method='mean')
        """
        
        if name not in self.predictors:
            print(f"✗ Predictor '{name}' not found")
            return None
        
        if self.bin_times is None:
            print("✗ Set bin times first with set_bin_times()")
            return None
        
        df = self.predictors[name].copy()
        
        if method == 'interpolate':
            # Interpolate to bin times
            f = interp1d(df['time'].values, df['value'].values, 
                        kind='linear', bounds_error=False, fill_value='extrapolate')
            binned_values = f(self.bin_times)
            binned_df = pd.DataFrame({'time': self.bin_times, 'value': binned_values})
        
        else:
            # Assign each time point to nearest bin
            df['bin_idx'] = np.digitize(df['time'], self.bin_times) - 1
            df['bin_idx'] = df['bin_idx'].clip(0, len(self.bin_times) - 1)
            df['bin_time'] = self.bin_times[df['bin_idx']]
            
            # Aggregate within bins
            if method == 'mean':
                binned_values = df.groupby('bin_time')['value'].mean()
            elif method == 'median':
                binned_values = df.groupby('bin_time')['value'].median()
            elif method == 'max':
                binned_values = df.groupby('bin_time')['value'].max()
            elif method == 'min':
                binned_values = df.groupby('bin_time')['value'].min()
            else:
                print(f"✗ Unknown method: {method}")
                return None
            
            binned_df = pd.DataFrame({
                'time': binned_values.index,
                'value': binned_values.values
            })
        
        if self.verbose:
            print(f"✓ Binned '{name}' using {method} method")
        
        return binned_df
    
    def rescale_predictor(self, name, method='minmax', log_transform=False):
        """
        Rescale predictor values for standardized interpretation.
        
        Parameters
        ----------
        name : str
            Predictor name
        method : str
            'minmax': scale to [0, 1]
            'zscore': standardize (mean=0, sd=1)
            'none': return as-is
        log_transform : bool
            Apply log transformation first (useful for positive-only data)
        
        Returns
        -------
        DataFrame
            Rescaled predictor
        
        Example
        -------
        >>> scaled_sl = proc.rescale_predictor('sea_level', method='minmax')
        """
        
        if name not in self.predictors:
            print(f"✗ Predictor '{name}' not found")
            return None
        
        df = self.predictors[name].copy()
        
        values = df['value'].values
        
        if log_transform:
            if np.any(values <= 0):
                print("⚠ Warning: log-transforming data with non-positive values")
                values = np.log(np.abs(values) + 1)
            else:
                values = np.log(values)
        
        if method == 'minmax':
            vmin, vmax = values.min(), values.max()
            values = (values - vmin) / (vmax - vmin)
            if self.verbose:
                print(f"✓ Rescaled '{name}' to [0, 1]")
        
        elif method == 'zscore':
            values = (values - values.mean()) / values.std()
            if self.verbose:
                print(f"✓ Z-score normalized '{name}'")
        
        df['value'] = values
        return df
    
    # ========================================================================
    # PART 3: EXPORT PREDICTOR FILES
    # ========================================================================
    
    def export_predictor(self, name, output_dir, filename=None, 
                        area='shared', binned=True, bin_method='mean'):
        """
        Export a single predictor to PyRateDES-compatible format.
        
        Format: tab-separated, columns: time (Ma) | value
        
        Parameters
        ----------
        name : str
            Predictor name
        output_dir : str
            Output directory
        filename : str
            Output filename. If None, uses predictor name
        area : str
            Area designation ('shared', 'area1', 'area2') - used in filename
        binned : bool
            If True, use binned values; else use raw
        bin_method : str
            Binning method if binned=True
        
        Returns
        -------
        str
            Path to exported file
        
        Example
        -------
        >>> path = proc.export_predictor('sea_level', './predictors', area='shared')
        """
        
        os.makedirs(output_dir, exist_ok=True)
        
        if binned and self.bin_times is not None:
            df = self.bin_predictor(name, method=bin_method)
        else:
            if name not in self.predictors:
                print(f"✗ Predictor '{name}' not found")
                return None
            df = self.predictors[name].copy()
        
        if filename is None:
            filename = f"{name}_{area}.txt"
        
        filepath = os.path.join(output_dir, filename)
        
        # Write with header
        with open(filepath, 'w') as f:
            f.write("time\tvalue\n")
            for _, row in df.iterrows():
                f.write(f"{row['time']:.4f}\t{row['value']:.6f}\n")
        
        if self.verbose:
            print(f"✓ Exported '{name}' to {filepath}")
        
        return filepath
    
    def export_for_dispersal(self, predictor_names, output_dir, bin_method='mean'):
        """
        Export multiple predictors as a folder for PyRateDES -varD.
        
        Parameters
        ----------
        predictor_names : list
            List of predictor names to export
        output_dir : str
            Output directory
        bin_method : str
            Binning method
        
        Returns
        -------
        str
            Path to folder containing exported files
        
        Example
        -------
        >>> proc.export_for_dispersal(['sea_level'], './des_predictors/dispersal')
        """
        
        folder = os.path.join(output_dir, 'dispersal')
        os.makedirs(folder, exist_ok=True)
        
        for pred_name in predictor_names:
            self.export_predictor(pred_name, folder, area='shared', 
                                 binned=True, bin_method=bin_method)
        
        if self.verbose:
            print(f"\n✓ Exported {len(predictor_names)} dispersal predictors")
            print(f"  Folder: {folder}")
        
        return folder
    
    def export_for_extinction(self, predictor_names, output_dir, bin_method='mean'):
        """
        Export multiple predictors for PyRateDES -varE.
        
        Parameters
        ----------
        predictor_names : list
            List of predictor names
        output_dir : str
            Output directory
        bin_method : str
            Binning method
        
        Returns
        -------
        str
            Path to folder
        
        Example
        -------
        >>> proc.export_for_extinction(['temperature'], './des_predictors/extinction')
        """
        
        folder = os.path.join(output_dir, 'extinction')
        os.makedirs(folder, exist_ok=True)
        
        for pred_name in predictor_names:
            self.export_predictor(pred_name, folder, area='shared', 
                                 binned=True, bin_method=bin_method)
        
        if self.verbose:
            print(f"\n✓ Exported {len(predictor_names)} extinction predictors")
            print(f"  Folder: {folder}")
        
        return folder
    
    # ========================================================================
    # PART 4: VISUALIZE PREDICTORS
    # ========================================================================
    
    def plot_predictors(self, names=None, figsize=(12, 6), save_path=None):
        """
        Plot predictor time series.
        
        Parameters
        ----------
        names : list
            Predictor names to plot. If None, plot all.
        figsize : tuple
            Figure size
        save_path : str
            If provided, save figure here
        
        Example
        -------
        >>> proc.plot_predictors(names=['sea_level', 'temperature'])
        """
        
        if names is None:
            names = list(self.predictors.keys())
        
        n_plots = len(names)
        fig, axes = plt.subplots(n_plots, 1, figsize=figsize)
        
        if n_plots == 1:
            axes = [axes]
        
        for ax, name in zip(axes, names):
            if name not in self.predictors:
                print(f"⚠ Predictor '{name}' not found")
                continue
            
            df = self.predictors[name]
            ax.plot(df['time'], df['value'], 'o-', linewidth=2, markersize=4, label=name)
            ax.set_xlabel('Time (Ma)')
            ax.set_ylabel('Value')
            ax.set_title(name)
            ax.grid(True, alpha=0.3)
            ax.invert_xaxis()  # Time increases to the left
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            if self.verbose:
                print(f"✓ Saved plot to {save_path}")
        
        return fig, axes
    
    def summary_statistics(self):
        """
        Print summary of all loaded predictors.
        """
        
        print("\n--- Predictor Summary ---\n")
        for name, df in self.predictors.items():
            print(f"{name}:")
            print(f"  Records: {len(df)}")
            print(f"  Time range: {df['time'].min():.2f} - {df['time'].max():.2f} Ma")
            print(f"  Value range: {df['value'].min():.4f} - {df['value'].max():.4f}")
            print()


# ============================================================================
# PART 5: CREATE SYNTHETIC PREDICTORS (for testing)
# ============================================================================

def create_synthetic_sea_level(t_max=50, n_points=500):
    """
    Create a synthetic sea level curve with realistic variability.
    
    Returns
    -------
    DataFrame
        Synthetic sea level (time, value)
    """
    
    time = np.linspace(0, t_max, n_points)
    
    # Base trend (mid-Cretaceous high → Cenozoic low)
    trend = -20 * (time / t_max) ** 1.5
    
    # Add periodic cycles (orbital forcing)
    cycles = 15 * np.sin(2 * np.pi * time / 5) + 10 * np.sin(2 * np.pi * time / 20)
    
    # Add noise
    noise = np.random.normal(0, 5, len(time))
    
    sea_level = trend + cycles + noise
    
    return pd.DataFrame({'time': time, 'value': sea_level})


def create_synthetic_temperature(t_max=50, n_points=500):
    """
    Create a synthetic temperature curve (cooler in recent times).
    
    Returns
    -------
    DataFrame
        Synthetic temperature (time, value in °C)
    """
    
    time = np.linspace(0, t_max, n_points)
    
    # Trend (Cenozoic cooling)
    trend = 20 + 5 * (time / t_max) ** 0.8
    
    # Variability (chaotic with some periodic component)
    noise = np.cumsum(np.random.normal(0, 0.5, len(time)))
    
    temp = trend + noise
    
    return pd.DataFrame({'time': time, 'value': temp})


# ============================================================================
# PART 6: EXAMPLE WORKFLOW
# ============================================================================

def example_workflow():
    """
    Complete example: create and export predictor files for DES
    """
    
    print("\n" + "="*70)
    print("PyRate DES Notebook 2: Predictor Preparation - Example Workflow")
    print("="*70 + "\n")
    
    # Initialize
    proc = PredictorProcessor(verbose=True)
    
    # Create synthetic data
    print("--- Step 1: Load Predictors ---")
    print("(Creating synthetic data...)\n")
    
    sea_level = create_synthetic_sea_level(t_max=50)
    temperature = create_synthetic_temperature(t_max=50)
    
    proc.predictors['sea_level'] = sea_level
    proc.predictors['temperature'] = temperature
    
    print("✓ Loaded synthetic sea level")
    print("✓ Loaded synthetic temperature")
    
    # Set bin times (to match DES input from Notebook 1)
    print("\n--- Step 2: Set Bin Times ---\n")
    bin_times = np.arange(0, 52.5, 2.5)[::-1]
    proc.set_bin_times(bin_times)
    
    # Bin predictors
    print("\n--- Step 3: Bin Predictors ---\n")
    binned_sl = proc.bin_predictor('sea_level', method='mean')
    binned_temp = proc.bin_predictor('temperature', method='mean')
    
    # Rescale
    print("\n--- Step 4: Rescale Predictors ---\n")
    proc.rescale_predictor('sea_level', method='minmax')
    proc.rescale_predictor('temperature', method='minmax')
    
    # Summary
    print("\n--- Step 5: Summary ---")
    proc.summary_statistics()
    
    # Export
    print("\n--- Step 6: Export for DES ---")
    output_dir = './example_des_predictors'
    os.makedirs(output_dir, exist_ok=True)
    
    proc.export_for_dispersal(['sea_level'], output_dir)
    proc.export_for_extinction(['temperature'], output_dir)
    
    # Plot
    print("\n--- Step 7: Plot ---\n")
    fig, axes = proc.plot_predictors(
        save_path=os.path.join(output_dir, 'predictors.png')
    )
    
    print(f"\n✓ Workflow complete!")
    print(f"  Output: {output_dir}/")
    print(f"  Ready for PyRateDES -varD and -varE flags!")
    
    return proc


# ============================================================================
# RUN EXAMPLE
# ============================================================================

if __name__ == '__main__':
    proc = example_workflow()
