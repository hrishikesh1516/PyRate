"""
================================================================================
PyRate DES Tutorial - Notebook 1: Prepare Fossil Data for DES Analysis
================================================================================

This notebook provides functions to:
1. Load PBDB (Paleobiology Database) CSV exports
2. Clean and filter occurrences
3. Assign geographic areas (discrete or from coordinates)
4. Create binned DES input files with age resampling (replicates)
5. Export as PyRateDES-compatible format

Author: PyRate community
Date: 2024
================================================================================
"""

import pandas as pd
import numpy as np
import os
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PART 1: LOAD AND EXPLORE PBDB DATA
# ============================================================================

class FossilDataProcessor:
    """
    Main class to handle PBDB data → DES input conversion
    """
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.data = None
        self.des_data = None
        self.binned_data = None
        
    def load_pbdb_csv(self, filepath, required_columns=None):
        """
        Load PBDB CSV export.
        
        Parameters
        ----------
        filepath : str
            Path to PBDB CSV file
        required_columns : list
            Must contain at minimum: scientificName, earliestAge, latestAge, 
            and either higherGeography (for discrete areas) or lat/lng 
            (for coordinate-based areas)
        
        Returns
        -------
        DataFrame
            Loaded and validated data
        
        Example
        -------
        >>> processor = FossilDataProcessor()
        >>> df = processor.load_pbdb_csv('my_fossils.csv')
        """
        
        if required_columns is None:
            required_columns = ['scientificName', 'earliestAge', 'latestAge', 'higherGeography']
        
        # Load
        try:
            df = pd.read_csv(filepath)
            if self.verbose:
                print(f"✓ Loaded {len(df)} records from {filepath}")
        except Exception as e:
            print(f"✗ Error loading file: {e}")
            return None
        
        # Check for required columns
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            print(f"✗ Missing required columns: {missing}")
            print(f"  Available columns: {list(df.columns)}")
            return None
        
        self.data = df
        
        # Summary
        if self.verbose:
            print(f"\n--- Data Summary ---")
            print(f"Total records: {len(df)}")
            print(f"Unique taxa: {df['scientificName'].nunique()}")
            print(f"Time span: {df['earliestAge'].max():.2f} - {df['latestAge'].min():.2f} Ma")
            print(f"Area column unique values: {df['higherGeography'].nunique() if 'higherGeography' in df.columns else 'N/A'}")
            if 'higherGeography' in df.columns:
                print(f"  Areas: {df['higherGeography'].unique().tolist()}")
        
        return df
    
    def filter_data(self, min_occurrences=1, remove_duplicates=False, age_range=None):
        """
        Filter fossil occurrences.
        
        Parameters
        ----------
        min_occurrences : int
            Minimum occurrences per taxon to retain
        remove_duplicates : bool
            If True, remove exact duplicates (same taxon, age, area)
        age_range : tuple
            (min_age, max_age) to retain. Example: (0, 100) for Cenozoic
        
        Returns
        -------
        DataFrame
            Filtered data
        
        Example
        -------
        >>> df_filtered = processor.filter_data(min_occurrences=3, age_range=(0, 66))
        """
        
        df = self.data.copy()
        n_start = len(df)
        
        # Remove missing area/age data
        df = df.dropna(subset=['earliestAge', 'latestAge', 'higherGeography'])
        
        if self.verbose and len(df) < n_start:
            print(f"Removed {n_start - len(df)} records with missing age/area data")
        
        # Age range filter
        if age_range is not None:
            df = df[(df['earliestAge'] <= age_range[1]) & (df['latestAge'] >= age_range[0])]
            if self.verbose:
                print(f"Kept records in age range {age_range[0]}-{age_range[1]} Ma: {len(df)}")
        
        # Remove duplicate occurrences (same taxon, age range, area)
        if remove_duplicates:
            n_pre_dup = len(df)
            df = df.drop_duplicates(
                subset=['scientificName', 'earliestAge', 'latestAge', 'higherGeography'],
                keep='first'
            )
            if self.verbose:
                print(f"Removed {n_pre_dup - len(df)} duplicate records")
        
        # Min occurrences per taxon
        taxon_counts = df['scientificName'].value_counts()
        df = df[df['scientificName'].isin(taxon_counts[taxon_counts >= min_occurrences].index)]
        
        if self.verbose:
            print(f"Retained taxa with ≥{min_occurrences} occurrences: {df['scientificName'].nunique()}")
        
        self.data = df
        return df
    
    def validate_areas(self, required_areas=2):
        """
        Check area coding and validate for DES (expects 2 discrete areas).
        
        Parameters
        ----------
        required_areas : int
            Expected number of distinct areas (default 2 for DES)
        
        Returns
        -------
        bool
            True if valid, False otherwise
        """
        
        unique_areas = self.data['higherGeography'].nunique()
        
        if unique_areas != required_areas:
            print(f"✗ Expected {required_areas} areas, but found {unique_areas}")
            print(f"  Areas: {self.data['higherGeography'].unique().tolist()}")
            return False
        
        if self.verbose:
            print(f"✓ Data has {unique_areas} valid areas")
        
        return True
    
    # ========================================================================
    # PART 2: ASSIGN AREAS (DISCRETE)
    # ========================================================================
    
    def assign_areas_discrete(self, area_mapping=None):
        """
        Assign numeric area codes (1, 2, 3 for A, B, A+B).
        
        Parameters
        ----------
        area_mapping : dict
            Mapping of area names to numeric codes.
            Example: {'Atlantic': 'AreaA', 'IndoPacific': 'AreaB'}
            If None, automatically uses first two unique values.
        
        Returns
        -------
        DataFrame
            Data with numeric 'area_code' column
        
        Example
        -------
        >>> mapping = {'Atlantic': 'Area1', 'IndoPacific': 'Area2'}
        >>> df = processor.assign_areas_discrete(area_mapping=mapping)
        """
        
        df = self.data.copy()
        
        if area_mapping is None:
            # Auto-assign: first two unique areas
            unique_areas = df['higherGeography'].unique()[:2]
            area_mapping = {area: f'Area{i+1}' for i, area in enumerate(unique_areas)}
        
        if self.verbose:
            print(f"Area mapping: {area_mapping}")
        
        # Map names to standardized area names
        df['area_name'] = df['higherGeography'].map(area_mapping)
        
        # Check for unmapped areas
        unmapped = df[df['area_name'].isna()]['higherGeography'].unique()
        if len(unmapped) > 0:
            print(f"✗ Unmapped areas: {unmapped}")
            return None
        
        # Convert to numeric codes for DES: Area1→1, Area2→2, Area1+Area2→3
        def get_area_code(row):
            areas = set(row['higherGeography'].split('|') if '|' in row['higherGeography'] else [row['higherGeography']])
            mapped_areas = {area_mapping[a] for a in areas if a in area_mapping}
            
            if mapped_areas == {'Area1'}:
                return 1
            elif mapped_areas == {'Area2'}:
                return 2
            elif mapped_areas == {'Area1', 'Area2'}:
                return 3
            else:
                return np.nan
        
        df['area_code'] = df.apply(get_area_code, axis=1)
        
        # Remove rows with NaN area codes
        df = df.dropna(subset=['area_code'])
        
        if self.verbose:
            print(f"✓ Assigned area codes to {len(df)} records")
            print(f"  Area 1: {(df['area_code']==1).sum()}")
            print(f"  Area 2: {(df['area_code']==2).sum()}")
            print(f"  Both areas: {(df['area_code']==3).sum()}")
        
        self.data = df
        return df
    
    # ========================================================================
    # PART 3: CREATE BINNED DES INPUT
    # ========================================================================
    
    def create_binned_input(self, bin_size=2.0, n_replicates=10, 
                           site_column=None, trim_age=None):
        """
        Create binned DES input files with age resampling (replicates).
        
        This function:
        1. Creates time bins
        2. For each replicate, randomly resamples fossil ages within their ranges
        3. Assigns each occurrence to a bin
        4. Groups occurrences by taxon and summarizes area occupancy per bin
        
        Parameters
        ----------
        bin_size : float
            Width of time bins (Myr). Default 2.0
        n_replicates : int
            Number of replicates (samples of age uncertainty). Default 10
        site_column : str
            Column name for site/assemblage ID. If provided, all occurrences 
            from same site are resampled together (coherent age draw)
        trim_age : float
            Maximum age to retain (older records removed)
        
        Returns
        -------
        dict
            Dictionary with keys:
            - 'binned_data': list of DataFrames (one per replicate)
            - 'time_series': array of bin boundaries (Ma)
            - 'bin_size': bin size used
        
        Example
        -------
        >>> result = processor.create_binned_input(bin_size=2.0, n_replicates=10)
        >>> binned_list = result['binned_data']
        >>> time_bins = result['time_series']
        """
        
        df = self.data.copy()
        
        # Trim age if specified
        if trim_age is not None:
            df = df[df['earliestAge'] <= trim_age]
            if self.verbose:
                print(f"Trimmed to max age {trim_age} Ma: {len(df)} records")
        
        # Create time bins (from oldest to youngest, reversed)
        max_age = df['earliestAge'].max()
        min_age = df['latestAge'].min()
        
        # Round to nearest bin_size
        max_age_rounded = np.ceil(max_age / bin_size) * bin_size
        min_age_rounded = np.floor(min_age / bin_size) * bin_size
        
        time_bins = np.arange(min_age_rounded, max_age_rounded + bin_size, bin_size)[::-1]
        
        if self.verbose:
            print(f"\n--- Binning Parameters ---")
            print(f"Bin size: {bin_size} Myr")
            print(f"Time range: {time_bins[-1]:.2f} - {time_bins[0]:.2f} Ma")
            print(f"Number of bins: {len(time_bins) - 1}")
        
        # Generate replicates
        binned_list = []
        
        for rep in range(n_replicates):
            if self.verbose and rep % 2 == 0:
                print(f"  Processing replicate {rep+1}/{n_replicates}...")
            
            df_rep = df.copy()
            
            # Resample ages
            if site_column is not None and site_column in df_rep.columns:
                # Coherent site resampling
                df_rep['age_resampled'] = np.nan
                for site in df_rep[site_column].unique():
                    site_mask = df_rep[site_column] == site
                    # Draw one age for this site
                    site_earliest = df_rep[site_mask]['earliestAge'].iloc[0]
                    site_latest = df_rep[site_mask]['latestAge'].iloc[0]
                    site_age = np.random.uniform(site_latest, site_earliest)
                    # Assign same age to all records from this site
                    df_rep.loc[site_mask, 'age_resampled'] = site_age
            else:
                # Independent resampling per occurrence
                df_rep['age_resampled'] = np.random.uniform(
                    df_rep['latestAge'].values,
                    df_rep['earliestAge'].values
                )
            
            # Assign to bins
            df_rep['bin_idx'] = np.digitize(df_rep['age_resampled'], time_bins) - 1
            df_rep['bin_idx'] = df_rep['bin_idx'].clip(0, len(time_bins) - 2)
            
            # Convert to bin time labels
            df_rep['bin_time'] = time_bins[df_rep['bin_idx']]
            
            # Summarize: for each taxon × bin, what area(s)?
            binned_rep = df_rep.groupby(['scientificName', 'bin_time'])['area_code'].apply(
                lambda x: list(x)
            ).reset_index()
            binned_rep.columns = ['scientificName', 'bin_time', 'areas']
            
            # Collapse to single area code per taxon × bin
            # If occurs in both areas within bin, code as 3
            binned_rep['area_summary'] = binned_rep['areas'].apply(
                lambda areas: 3 if set(areas) == {1, 2} 
                              else (1 if 1 in areas else (2 if 2 in areas else np.nan))
            )
            
            # Pivot to wide format: rows = taxa, columns = bins
            pivot = binned_rep.pivot_table(
                index='scientificName',
                columns='bin_time',
                values='area_summary'
            )
            
            # Ensure all bins are present (fill missing with NaN)
            for bin_time in time_bins[:-1]:
                if bin_time not in pivot.columns:
                    pivot[bin_time] = np.nan
            
            # Sort columns by age (oldest to youngest)
            pivot = pivot[sorted(pivot.columns, reverse=True)]
            
            binned_list.append(pivot)
        
        self.binned_data = binned_list
        
        if self.verbose:
            print(f"✓ Created {n_replicates} binned replicates")
            print(f"  Average taxa per replicate: {np.mean([len(b) for b in binned_list]):.0f}")
        
        return {
            'binned_data': binned_list,
            'time_series': time_bins[:-1],  # Bin midpoints
            'bin_size': bin_size,
            'n_replicates': n_replicates
        }
    
    # ========================================================================
    # PART 4: EXPORT TO DES FORMAT
    # ========================================================================
    
    def export_des_input(self, output_dir, prefix='DES_input'):
        """
        Export binned data to PyRateDES-compatible format.
        
        Generates files: {prefix}_1.txt, {prefix}_2.txt, ...
        Format: tab-separated, rows = taxa, columns = time bins (oldest to youngest)
        Values: 1 (area A), 2 (area B), 3 (both), NaN (absent)
        
        Parameters
        ----------
        output_dir : str
            Directory to save files
        prefix : str
            Filename prefix
        
        Returns
        -------
        list
            Paths to exported files
        
        Example
        -------
        >>> paths = processor.export_des_input('./des_inputs', prefix='Bivalves')
        """
        
        if self.binned_data is None:
            print("✗ No binned data to export. Run create_binned_input() first.")
            return None
        
        os.makedirs(output_dir, exist_ok=True)
        
        output_paths = []
        
        for rep_idx, df in enumerate(self.binned_data):
            filename = f"{prefix}_{rep_idx + 1}.txt"
            filepath = os.path.join(output_dir, filename)
            
            # Reset index to make scientificName a column
            df_export = df.reset_index()
            df_export.columns.name = None  # Remove column name
            
            # Write with header
            with open(filepath, 'w') as f:
                # Header: tab-separated bin times
                header = 'scientificName\t' + '\t'.join([f'bin_{t:.2f}' for t in df.columns])
                f.write(header + '\n')
                
                # Data: replace NaN with empty string for cleaner output
                for idx, row in df_export.iterrows():
                    taxon = row['scientificName']
                    values = []
                    for bin_time in df.columns:
                        val = row[bin_time]
                        if pd.isna(val):
                            values.append('')
                        else:
                            values.append(str(int(val)))
                    line = taxon + '\t' + '\t'.join(values)
                    f.write(line + '\n')
            
            output_paths.append(filepath)
            
            if self.verbose:
                print(f"✓ Exported {filepath}")
        
        if self.verbose:
            print(f"\n✓ Exported {len(output_paths)} DES input files to {output_dir}")
        
        return output_paths
    
    def export_summary_stats(self, output_file):
        """
        Export summary statistics to text file.
        
        Parameters
        ----------
        output_file : str
            Path to output file
        """
        
        df = self.data
        
        summary = f"""
DES Input Data Summary
======================

GENERAL STATISTICS
-------------------
Total occurrences: {len(df)}
Unique taxa: {df['scientificName'].nunique()}
Time span: {df['earliestAge'].max():.2f} - {df['latestAge'].min():.2f} Ma

AREA DISTRIBUTION
-------------------
Area 1 only: {(df['area_code']==1).sum()}
Area 2 only: {(df['area_code']==2).sum()}
Both areas: {(df['area_code']==3).sum()}

TAXON-LEVEL STATISTICS
------------------------
Mean occurrences per taxon: {len(df) / df['scientificName'].nunique():.2f}
Min occurrences per taxon: {df['scientificName'].value_counts().min()}
Max occurrences per taxon: {df['scientificName'].value_counts().max()}

BINNING PARAMETERS USED
------------------------
(See DES input files for details)
"""
        
        with open(output_file, 'w') as f:
            f.write(summary)
        
        if self.verbose:
            print(f"✓ Summary statistics exported to {output_file}")


# ============================================================================
# PART 5: EXAMPLE WORKFLOW
# ============================================================================

def example_workflow():
    """
    Complete example: PBDB CSV → DES input files
    """
    
    print("\n" + "="*70)
    print("PyRate DES Notebook 1: Data Preparation - Example Workflow")
    print("="*70 + "\n")
    
    # Initialize processor
    processor = FossilDataProcessor(verbose=True)
    
    # Example 1: Load and filter
    print("\n--- Step 1: Load PBDB Data ---")
    # Note: Replace with your actual file path
    # df = processor.load_pbdb_csv('your_pbdb_export.csv')
    
    # For this example, create synthetic data
    print("(Creating synthetic example data...)\n")
    
    np.random.seed(42)
    n_records = 200
    
    synthetic_data = pd.DataFrame({
        'scientificName': np.random.choice(
            [f'Taxon_{i}' for i in range(30)], n_records
        ),
        'earliestAge': np.random.uniform(20, 50, n_records),
        'latestAge': np.random.uniform(0, 20, n_records),
        'higherGeography': np.random.choice(['Atlantic', 'IndoPacific'], n_records)
    })
    
    # Fix age order (earliest > latest)
    synthetic_data[['earliestAge', 'latestAge']] = synthetic_data[
        ['earliestAge', 'latestAge']
    ].apply(lambda x: sorted(x, reverse=True), axis=1, result_type='expand')
    
    processor.data = synthetic_data
    
    if processor.verbose:
        print(f"✓ Created synthetic dataset with {len(synthetic_data)} records")
        print(f"  Unique taxa: {synthetic_data['scientificName'].nunique()}")
        print(f"  Areas: {synthetic_data['higherGeography'].unique().tolist()}")
    
    # Step 2: Filter
    print("\n--- Step 2: Filter Data ---")
    processor.filter_data(min_occurrences=2)
    
    # Step 3: Assign areas
    print("\n--- Step 3: Assign Discrete Areas ---")
    processor.assign_areas_discrete()
    
    # Step 4: Create binned input
    print("\n--- Step 4: Create Binned DES Input ---")
    result = processor.create_binned_input(bin_size=2.5, n_replicates=5)
    
    # Step 5: Export
    print("\n--- Step 5: Export to DES Format ---")
    output_dir = './example_des_inputs'
    paths = processor.export_des_input(output_dir, prefix='Example')
    
    # Step 6: Summary
    print("\n--- Step 6: Summary Statistics ---")
    processor.export_summary_stats(os.path.join(output_dir, 'summary.txt'))
    
    print(f"\n✓ Workflow complete!")
    print(f"  Output files: {output_dir}/")
    print(f"  Ready for PyRateDES analysis!")
    
    return processor, result


# ============================================================================
# RUN EXAMPLE
# ============================================================================

if __name__ == '__main__':
    processor, result = example_workflow()
    
    # Print example of first binned replicate
    print("\n--- Example: First Binned Replicate ---")
    print(result['binned_data'][0].head(10))
    print(f"\nShape: {result['binned_data'][0].shape}")
    print(f"(rows = taxa, columns = time bins)")
