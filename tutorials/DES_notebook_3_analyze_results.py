"""
================================================================================
PyRate DES Tutorial - Notebook 3: Analyze and Compare DES Results
================================================================================

This notebook provides functions to:
1. Load and parse PyRateDES .log files (MCMC traces)
2. Compute posterior statistics (means, credible intervals, HPD)
3. Compare models (likelihood, AIC, Bayes factors)
4. Analyze parameter estimates and their uncertainties
5. Visualize rate trajectories and covariate effects
6. Export summary tables

Author: PyRate community
Date: 2024
================================================================================
"""

import pandas as pd
import numpy as np
import os
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# PART 1: LOAD AND PARSE DES LOG FILES
# ============================================================================

class DESResultAnalyzer:
    """
    Analyze and compare PyRateDES model results.
    """
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.logs = {}  # dict: model_name → DataFrame
        self.summaries = {}  # dict: model_name → summary dict
        self.marginal_rates = {}  # dict: model_name → DataFrame
    
    def load_mcmc_log(self, model_name, filepath, burnin_fraction=0.2):
        """
        Load MCMC log file and apply burn-in.
        
        Parameters
        ----------
        model_name : str
            Name for this model (e.g., 'constant', 'varD', 'DivdE')
        filepath : str
            Path to *_mcmc.log file
        burnin_fraction : float
            Fraction of samples to discard as burn-in (0.2 = 20%)
        
        Returns
        -------
        DataFrame
            MCMC trace after burn-in
        
        Example
        -------
        >>> analyzer = DESResultAnalyzer()
        >>> log = analyzer.load_mcmc_log('constant', 'Bivalves_0.log', burnin_fraction=0.1)
        """
        
        try:
            df = pd.read_csv(filepath, sep='\t')
        except Exception as e:
            print(f"✗ Error loading {filepath}: {e}")
            return None
        
        # Apply burn-in (discard first N% of samples)
        burnin_idx = int(len(df) * burnin_fraction)
        df_post = df.iloc[burnin_idx:].reset_index(drop=True)
        
        self.logs[model_name] = df_post
        
        if self.verbose:
            print(f"✓ Loaded '{model_name}'")
            print(f"  Total iterations: {len(df)}")
            print(f"  Post-burnin samples: {len(df_post)}")
            if 'likelihood' in df_post.columns:
                print(f"  Likelihood range: {df_post['likelihood'].min():.2f} - {df_post['likelihood'].max():.2f}")
        
        return df_post
    
    def get_parameter_columns(self, model_name, pattern=None):
        """
        Extract parameter column names from log.
        
        Parameters
        ----------
        model_name : str
            Model name
        pattern : str
            Filter columns matching pattern (e.g., 'd12_' for dispersal d12)
        
        Returns
        -------
        list
            Matching column names
        
        Example
        -------
        >>> d12_cols = analyzer.get_parameter_columns('varD', pattern='d12_')
        """
        
        if model_name not in self.logs:
            print(f"✗ Model '{model_name}' not loaded")
            return None
        
        cols = list(self.logs[model_name].columns)
        
        # Filter out standard columns
        standard = ['it', 'posterior', 'prior', 'likelihood']
        param_cols = [c for c in cols if c not in standard]
        
        if pattern:
            param_cols = [c for c in param_cols if pattern in c]
        
        return param_cols
    
    # ========================================================================
    # PART 2: COMPUTE POSTERIOR STATISTICS
    # ========================================================================
    
    def hpd(self, data, credibility=0.95):
        """
        Compute highest posterior density (HPD) interval.
        
        Parameters
        ----------
        data : array-like
            Posterior samples
        credibility : float
            Credibility level (0.95 = 95% HPD)
        
        Returns
        -------
        tuple
            (lower_bound, upper_bound)
        """
        
        data = np.asarray(data)
        data = np.sort(data)
        n = len(data)
        interval_idx_width = int(np.ceil(credibility * n))
        n_intervals = n - interval_idx_width
        
        if n_intervals < 1:
            return data.min(), data.max()
        
        intervals = np.zeros(n_intervals)
        for i in range(n_intervals):
            intervals[i] = data[i + interval_idx_width] - data[i]
        
        min_idx = np.argmin(intervals)
        hpd_min = data[min_idx]
        hpd_max = data[min_idx + interval_idx_width]
        
        return hpd_min, hpd_max
    
    def summarize_parameter(self, model_name, param_name, credibility=0.95):
        """
        Compute summary statistics for a single parameter.
        
        Parameters
        ----------
        model_name : str
            Model name
        param_name : str
            Parameter column name
        credibility : float
            Credibility level for HPD
        
        Returns
        -------
        dict
            Statistics: mean, median, sd, HPD_lower, HPD_upper, ESS
        
        Example
        -------
        >>> stats = analyzer.summarize_parameter('constant', 'd12_t25.0')
        """
        
        if model_name not in self.logs:
            print(f"✗ Model '{model_name}' not loaded")
            return None
        
        if param_name not in self.logs[model_name].columns:
            print(f"✗ Parameter '{param_name}' not found")
            return None
        
        data = self.logs[model_name][param_name].values
        
        mean = np.mean(data)
        median = np.median(data)
        sd = np.std(data)
        hpd_low, hpd_high = self.hpd(data, credibility)
        
        # Effective sample size (simple autocorrelation method)
        ess = len(data) / (1 + 2 * self._acf_lag1(data))
        
        return {
            'mean': mean,
            'median': median,
            'sd': sd,
            'HPD_lower': hpd_low,
            'HPD_upper': hpd_high,
            'ESS': ess,
            'n_samples': len(data)
        }
    
    def _acf_lag1(self, data):
        """Autocorrelation at lag 1."""
        c0 = np.mean((data - np.mean(data)) ** 2)
        c1 = np.mean((data[:-1] - np.mean(data)) * (data[1:] - np.mean(data)))
        return c1 / c0 if c0 > 0 else 0
    
    def summarize_all_parameters(self, model_name, credibility=0.95):
        """
        Summarize all parameters in a model.
        
        Parameters
        ----------
        model_name : str
            Model name
        credibility : float
            Credibility level
        
        Returns
        -------
        DataFrame
            Summary table for all parameters
        
        Example
        -------
        >>> summary = analyzer.summarize_all_parameters('varD')
        """
        
        param_cols = self.get_parameter_columns(model_name)
        
        summaries = []
        for param in param_cols:
            param_stats = self.summarize_parameter(model_name, param, credibility)
            if param_stats:
                param_stats['parameter'] = param
                summaries.append(param_stats)
        
        df_summary = pd.DataFrame(summaries)
        self.summaries[model_name] = df_summary
        
        if self.verbose:
            print(f"✓ Summarized {len(df_summary)} parameters for '{model_name}'")
        
        return df_summary
    
    # ========================================================================
    # PART 3: MODEL COMPARISON
    # ========================================================================
    
    def get_model_likelihood(self, model_name, statistic='mean'):
        """
        Get mean or median likelihood of a model.
        
        Parameters
        ----------
        model_name : str
            Model name
        statistic : str
            'mean', 'median', or 'max'
        
        Returns
        -------
        float
            Mean/median/max likelihood
        """
        
        if model_name not in self.logs:
            print(f"✗ Model '{model_name}' not loaded")
            return None
        
        if 'likelihood' not in self.logs[model_name].columns:
            print(f"⚠ No 'likelihood' column in {model_name}")
            return None
        
        lik = self.logs[model_name]['likelihood'].values
        
        if statistic == 'mean':
            return np.mean(lik)
        elif statistic == 'median':
            return np.median(lik)
        else:
            return np.max(lik)
    
    def compute_aic(self, model_name, n_parameters, n_taxa):
        """
        Compute AIC for model selection.
        
        Parameters
        ----------
        model_name : str
            Model name
        n_parameters : int
            Number of free parameters in the model
        n_taxa : int
            Number of taxa (used for AICc calculation)
        
        Returns
        -------
        dict
            AIC, AICc, and related statistics
        
        Example
        -------
        >>> aic = analyzer.compute_aic('constant', n_parameters=6, n_taxa=50)
        """
        
        if 'likelihood' not in self.logs[model_name].columns:
            print(f"✗ No likelihood column in {model_name}")
            return None
        
        max_lik = self.logs[model_name]['likelihood'].max()
        
        aic = 2 * n_parameters - 2 * max_lik
        
        # AICc (small sample correction)
        denom = n_taxa - n_parameters - 1
        aicc = np.inf if denom <= 0 else aic + (2 * n_parameters**2 + 2 * n_parameters) / denom
        
        return {
            'model': model_name,
            'max_likelihood': max_lik,
            'n_parameters': n_parameters,
            'AIC': aic,
            'AICc': aicc
        }
    
    def compare_models_aic(self, model_comparisons):
        """
        Compare multiple models using AIC weights.
        
        Parameters
        ----------
        model_comparisons : list of dict
            Each dict: {'name': str, 'n_parameters': int, 'n_taxa': int}
        
        Returns
        -------
        DataFrame
            Model comparison table with AIC weights and relative likelihoods
        
        Example
        -------
        >>> comparisons = [
        ...     {'name': 'constant', 'n_parameters': 6, 'n_taxa': 50},
        ...     {'name': 'varD', 'n_parameters': 8, 'n_taxa': 50},
        ... ]
        >>> df_comp = analyzer.compare_models_aic(comparisons)
        """
        
        aic_list = []
        for comp in model_comparisons:
            aic_dict = self.compute_aic(comp['name'], comp['n_parameters'], comp['n_taxa'])
            if aic_dict:
                aic_list.append(aic_dict)
        
        df = pd.DataFrame(aic_list)
        
        # Compute AIC weights
        min_aic = df['AIC'].min()
        df['delta_AIC'] = df['AIC'] - min_aic
        df['AIC_weight'] = np.exp(-0.5 * df['delta_AIC'])
        df['AIC_weight'] = df['AIC_weight'] / df['AIC_weight'].sum()
        
        # Relative likelihood
        df['relative_likelihood'] = np.exp(-0.5 * df['delta_AIC'])
        
        df = df.sort_values('AIC_weight', ascending=False)
        
        if self.verbose:
            print(f"\n--- Model Comparison (AIC) ---\n")
            cols_to_show = ['model', 'n_parameters', 'AIC', 'delta_AIC', 'AIC_weight']
            print(df[cols_to_show].to_string(index=False))
        
        return df
    
    # ========================================================================
    # PART 4: VISUALIZE RESULTS
    # ========================================================================
    
    def plot_trace(self, model_name, param_names=None, figsize=(12, 8), save_path=None):
        """
        Plot MCMC trace plots.
        
        Parameters
        ----------
        model_name : str
            Model name
        param_names : list
            Parameter names to plot. If None, plot first 6 parameters
        figsize : tuple
            Figure size
        save_path : str
            If provided, save figure
        
        Example
        -------
        >>> analyzer.plot_trace('constant', param_names=['d12_t25.0', 'e1_t25.0'])
        """
        
        if model_name not in self.logs:
            print(f"✗ Model '{model_name}' not loaded")
            return None
        
        if param_names is None:
            param_names = self.get_parameter_columns(model_name)[:6]
        
        n_params = len(param_names)
        fig, axes = plt.subplots(n_params, 1, figsize=figsize)
        
        if n_params == 1:
            axes = [axes]
        
        df = self.logs[model_name]
        
        for ax, param in zip(axes, param_names):
            if param in df.columns:
                trace = df[param].values
                ax.plot(trace, linewidth=0.5, alpha=0.7)
                ax.set_ylabel(param)
                ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Iteration')
        fig.suptitle(f"MCMC Trace: {model_name}")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            if self.verbose:
                print(f"✓ Saved trace plot to {save_path}")
        
        return fig, axes
    
    def plot_posterior_distribution(self, model_name, param_names=None, figsize=(12, 8), save_path=None):
        """
        Plot posterior distributions (histograms + KDE).
        
        Parameters
        ----------
        model_name : str
            Model name
        param_names : list
            Parameters to plot
        figsize : tuple
            Figure size
        save_path : str
            Save path
        
        Example
        -------
        >>> analyzer.plot_posterior_distribution('constant', param_names=['d12_t25.0', 'e1_t25.0'])
        """
        
        if param_names is None:
            param_names = self.get_parameter_columns(model_name)[:6]
        
        n_params = len(param_names)
        fig, axes = plt.subplots(n_params, 1, figsize=figsize)
        
        if n_params == 1:
            axes = [axes]
        
        df = self.logs[model_name]
        
        for ax, param in zip(axes, param_names):
            if param not in df.columns:
                continue
            
            data = df[param].values
            ax.hist(data, bins=50, density=True, alpha=0.6, label='Posterior', color='steelblue')
            
            # KDE
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 200)
            ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
            
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.set_title(param)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        fig.suptitle(f"Posterior Distributions: {model_name}")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            if self.verbose:
                print(f"✓ Saved posterior plot to {save_path}")
        
        return fig, axes
    
    # ========================================================================
    # PART 5: EXPORT SUMMARY TABLES
    # ========================================================================
    
    def export_summary_table(self, model_name, output_file):
        """
        Export parameter summary table to CSV.
        
        Parameters
        ----------
        model_name : str
            Model name
        output_file : str
            Output CSV path
        
        Example
        -------
        >>> analyzer.export_summary_table('varD', 'varD_summary.csv')
        """
        
        if model_name not in self.summaries:
            print(f"Summarizing {model_name}...")
            self.summarize_all_parameters(model_name)
        
        self.summaries[model_name].to_csv(output_file, index=False)
        
        if self.verbose:
            print(f"✓ Exported summary to {output_file}")
    
    def export_comparison_table(self, comparison_results, output_file):
        """
        Export model comparison results.
        
        Parameters
        ----------
        comparison_results : DataFrame
            From compare_models_aic()
        output_file : str
            Output path
        """
        
        comparison_results.to_csv(output_file, index=False)
        
        if self.verbose:
            print(f"✓ Exported comparison table to {output_file}")


# ============================================================================
# PART 6: EXAMPLE WORKFLOW
# ============================================================================

def create_synthetic_log(n_samples=1000):
    """
    Create synthetic MCMC log for testing.
    """
    
    iterations = np.arange(n_samples)
    
    # Simulate parameters with random walk
    d12_t25 = 0.15 + np.cumsum(np.random.normal(0, 0.01, n_samples))
    d21_t25 = 0.12 + np.cumsum(np.random.normal(0, 0.015, n_samples))
    e1_t25 = 0.05 + np.cumsum(np.random.normal(0, 0.005, n_samples))
    e2_t25 = 0.03 + np.cumsum(np.random.normal(0, 0.003, n_samples))
    
    # Likelihood
    likelihood = -(d12_t25**2 + d21_t25**2 + e1_t25**2 + e2_t25**2)
    prior = np.random.normal(-1, 0.5, n_samples)
    posterior = likelihood + prior
    
    df = pd.DataFrame({
        'it': iterations,
        'posterior': posterior,
        'prior': prior,
        'likelihood': likelihood,
        'd12_t25.0': d12_t25,
        'd21_t25.0': d21_t25,
        'e1_t25.0': e1_t25,
        'e2_t25.0': e2_t25
    })
    
    return df


def example_workflow():
    """
    Complete example: load, analyze, and compare DES results
    """
    
    print("\n" + "="*70)
    print("PyRate DES Notebook 3: Results Analysis - Example Workflow")
    print("="*70 + "\n")
    
    # Initialize
    analyzer = DESResultAnalyzer(verbose=True)
    
    # Create synthetic logs
    print("--- Step 1: Load Logs ---\n")
    
    # For demo, create synthetic data
    log_const = create_synthetic_log(n_samples=2000)
    log_vard = create_synthetic_log(n_samples=2000) + np.random.normal(0, 0.05, 2000)
    
    # Save to temp files
    temp_dir = './temp_des_logs'
    os.makedirs(temp_dir, exist_ok=True)
    
    log_const.to_csv(f'{temp_dir}/constant.log', sep='\t', index=False)
    log_vard.to_csv(f'{temp_dir}/varD.log', sep='\t', index=False)
    
    # Load
    analyzer.load_mcmc_log('constant', f'{temp_dir}/constant.log', burnin_fraction=0.2)
    analyzer.load_mcmc_log('varD', f'{temp_dir}/varD.log', burnin_fraction=0.2)
    
    # Summarize
    print("\n--- Step 2: Summarize Parameters ---\n")
    
    summary_const = analyzer.summarize_all_parameters('constant')
    summary_vard = analyzer.summarize_all_parameters('varD')
    
    print("\nConstant model summary (first 5 parameters):")
    print(summary_const[['parameter', 'mean', 'sd', 'HPD_lower', 'HPD_upper']].head())
    
    # Compare models
    print("\n--- Step 3: Compare Models ---\n")
    
    comparisons = [
        {'name': 'constant', 'n_parameters': 6, 'n_taxa': 50},
        {'name': 'varD', 'n_parameters': 8, 'n_taxa': 50}
    ]
    
    df_comp = analyzer.compare_models_aic(comparisons)
    
    # Plot traces
    print("\n--- Step 4: Plots ---\n")
    
    fig, ax = analyzer.plot_trace('constant', figsize=(10, 6), 
                                  save_path=f'{temp_dir}/trace_constant.png')
    print(f"✓ Saved trace plot")
    
    fig, ax = analyzer.plot_posterior_distribution('constant', figsize=(10, 8),
                                                   save_path=f'{temp_dir}/posterior_constant.png')
    print(f"✓ Saved posterior plot")
    
    # Export summaries
    print("\n--- Step 5: Export ---\n")
    
    analyzer.export_summary_table('constant', f'{temp_dir}/summary_constant.csv')
    analyzer.export_comparison_table(df_comp, f'{temp_dir}/model_comparison.csv')
    
    print(f"\n✓ Workflow complete!")
    print(f"  Output: {temp_dir}/")
    
    return analyzer


# ============================================================================
# RUN EXAMPLE
# ============================================================================

if __name__ == '__main__':
    analyzer = example_workflow()
