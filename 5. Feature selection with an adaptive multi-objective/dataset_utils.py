"""
Dataset Utilities for MOBGA-AOS Experiments
Handles loading, preprocessing, and validation of assignment datasets

Datasets: DS02, DS04, DS05, DS07, DS08, DS10
"""

import numpy as np
import os
import zipfile


# Dataset information based on assignment and original paper
DATASET_INFO = {
    'DS02': {
        'name': 'LungCancer',
        'n_features': 56,
        'n_classes': 3,
        'n_instances': 32,
        'description': 'Lung Cancer dataset - small samples, moderate features'
    },
    'DS04': {
        'name': 'OpticalRecognitionofHandwritten',
        'n_features': 64,
        'n_classes': 10,
        'n_instances': 1000,
        'description': 'Optical Recognition of Handwritten Digits'
    },
    'DS05': {
        'name': 'MadelonValid',
        'n_features': 500,
        'n_classes': 2,
        'n_instances': 600,
        'description': 'MADELON - artificial dataset with many noisy features'
    },
    'DS07': {
        'name': 'Har',
        'n_features': 561,
        'n_classes': 6,
        'n_instances': 900,
        'description': 'Human Activity Recognition using Smartphones'
    },
    'DS08': {
        'name': 'HAPT',
        'n_features': 561,
        'n_classes': 12,
        'n_instances': 1200,
        'description': 'Human Activity Prediction with Transitions'
    },
    'DS10': {
        'name': 'MultipleFeaturesDigit',
        'n_features': 649,
        'n_classes': 10,
        'n_instances': 1000,
        'description': 'Multiple Features Dataset - Handwritten Digits'
    }
}


def extract_datasets(zip_path='Datasets.zip', output_dir='.'):
    """
    Extract datasets from the provided zip file.
    
    Parameters:
    -----------
    zip_path : str - Path to the Datasets.zip file
    output_dir : str - Directory to extract files to
    
    Returns:
    --------
    list - List of extracted file paths
    """
    if not os.path.exists(zip_path):
        print(f"Error: {zip_path} not found")
        return []
    
    extracted_files = []
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            if file_info.filename.endswith('.csv'):
                zip_ref.extract(file_info, output_dir)
                extracted_files.append(os.path.join(output_dir, file_info.filename))
                print(f"Extracted: {file_info.filename}")
    
    return extracted_files


def load_dataset(filepath):
    """
    Load dataset from CSV file.
    Assumes last column is the target variable.
    
    Parameters:
    -----------
    filepath : str - Path to CSV file
    
    Returns:
    --------
    tuple - (X, y) where X is feature matrix and y is label vector
    """
    try:
        # Try loading without header first
        data = np.genfromtxt(filepath, delimiter=',', skip_header=0)
        
        # Check for NaN values (indicates header row)
        if np.isnan(data).any():
            data = np.genfromtxt(filepath, delimiter=',', skip_header=1)
        
        # Separate features and labels
        X = data[:, :-1]
        y = data[:, -1].astype(int)
        
        return X, y
        
    except Exception as e:
        raise ValueError(f"Error loading {filepath}: {e}")


def validate_dataset(filepath, expected_info=None):
    """
    Validate a loaded dataset against expected specifications.
    
    Parameters:
    -----------
    filepath : str - Path to dataset file
    expected_info : dict - Expected dataset information (optional)
    
    Returns:
    --------
    dict - Validation results
    """
    try:
        X, y = load_dataset(filepath)
        
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        results = {
            'valid': True,
            'filepath': filepath,
            'n_samples': n_samples,
            'n_features': n_features,
            'n_classes': n_classes,
            'has_nan': np.isnan(X).any(),
            'has_inf': np.isinf(X).any(),
            'class_distribution': dict(zip(*np.unique(y, return_counts=True))),
            'feature_range': (float(X.min()), float(X.max())),
        }
        
        # Compare with expected info if provided
        if expected_info:
            if expected_info.get('n_features') and n_features != expected_info['n_features']:
                results['warning'] = f"Expected {expected_info['n_features']} features, got {n_features}"
            if expected_info.get('n_classes') and n_classes != expected_info['n_classes']:
                results['warning'] = f"Expected {expected_info['n_classes']} classes, got {n_classes}"
        
        return results
        
    except Exception as e:
        return {
            'valid': False,
            'filepath': filepath,
            'error': str(e)
        }


def normalize_data(X, method='minmax'):
    """
    Normalize feature matrix.
    
    Parameters:
    -----------
    X : np.ndarray - Feature matrix
    method : str - 'minmax' for [0,1] scaling, 'zscore' for standardization
    
    Returns:
    --------
    np.ndarray - Normalized feature matrix
    """
    if method == 'minmax':
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)
        denom = X_max - X_min
        denom[denom == 0] = 1  # Avoid division by zero
        return (X - X_min) / denom
    
    elif method == 'zscore':
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1  # Avoid division by zero
        return (X - mean) / std
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def train_test_split(X, y, train_ratio=0.7, seed=42, stratified=False):
    """
    Split data into training and test sets.
    
    Parameters:
    -----------
    X : np.ndarray - Feature matrix
    y : np.ndarray - Labels
    train_ratio : float - Proportion of data for training
    seed : int - Random seed
    stratified : bool - Whether to maintain class proportions
    
    Returns:
    --------
    tuple - (X_train, y_train, X_test, y_test)
    """
    np.random.seed(seed)
    n_samples = X.shape[0]
    
    if stratified:
        # Stratified split
        train_indices = []
        test_indices = []
        
        for cls in np.unique(y):
            cls_indices = np.where(y == cls)[0]
            np.random.shuffle(cls_indices)
            n_train = int(len(cls_indices) * train_ratio)
            train_indices.extend(cls_indices[:n_train])
            test_indices.extend(cls_indices[n_train:])
        
        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)
        
    else:
        # Random split
        indices = np.random.permutation(n_samples)
        train_size = int(n_samples * train_ratio)
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
    
    return X[train_indices], y[train_indices], X[test_indices], y[test_indices]


def get_dataset_stats(X, y):
    """
    Get comprehensive statistics about a dataset.
    
    Parameters:
    -----------
    X : np.ndarray - Feature matrix
    y : np.ndarray - Labels
    
    Returns:
    --------
    dict - Dataset statistics
    """
    return {
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'n_classes': len(np.unique(y)),
        'class_counts': dict(zip(*np.unique(y, return_counts=True))),
        'feature_means': X.mean(axis=0).tolist(),
        'feature_stds': X.std(axis=0).tolist(),
        'feature_mins': X.min(axis=0).tolist(),
        'feature_maxs': X.max(axis=0).tolist(),
        'has_missing': np.isnan(X).any(),
        'sparsity': (X == 0).mean(),
    }


def print_dataset_summary(filepath):
    """
    Print a summary of dataset properties.
    
    Parameters:
    -----------
    filepath : str - Path to dataset file
    """
    try:
        X, y = load_dataset(filepath)
        stats = get_dataset_stats(X, y)
        
        basename = os.path.basename(filepath)
        ds_id = basename.replace('.csv', '')
        
        print(f"\n{'='*50}")
        print(f"Dataset: {ds_id}")
        print(f"{'='*50}")
        print(f"Samples:  {stats['n_samples']}")
        print(f"Features: {stats['n_features']}")
        print(f"Classes:  {stats['n_classes']}")
        print(f"Class distribution: {stats['class_counts']}")
        print(f"Missing values: {stats['has_missing']}")
        print(f"Sparsity: {stats['sparsity']:.2%}")
        print(f"Feature range: [{min(stats['feature_mins']):.2f}, {max(stats['feature_maxs']):.2f}]")
        
        # Check against expected info
        if ds_id in DATASET_INFO:
            info = DATASET_INFO[ds_id]
            print(f"\nExpected: {info['n_features']} features, {info['n_classes']} classes")
            if stats['n_features'] != info['n_features']:
                print(f"WARNING: Feature count mismatch!")
        
    except Exception as e:
        print(f"Error loading {filepath}: {e}")


def prepare_all_datasets(data_dir='.', zip_path='Datasets.zip'):
    """
    Prepare and validate all datasets for experiments.
    
    Parameters:
    -----------
    data_dir : str - Directory containing dataset files
    zip_path : str - Path to Datasets.zip (if needs extraction)
    
    Returns:
    --------
    dict - Dictionary of {ds_id: (X, y)} for all valid datasets
    """
    datasets = {}
    
    # Check if we need to extract
    expected_files = ['DS02.csv', 'DS04.csv', 'DS05.csv', 'DS07.csv', 'DS08.csv', 'DS10.csv']
    missing = [f for f in expected_files if not os.path.exists(os.path.join(data_dir, f))]
    
    if missing and os.path.exists(zip_path):
        print(f"Extracting datasets from {zip_path}...")
        extract_datasets(zip_path, data_dir)
    
    # Load each dataset
    for ds_id in ['DS02', 'DS04', 'DS05', 'DS07', 'DS08', 'DS10']:
        filepath = os.path.join(data_dir, f'{ds_id}.csv')
        
        if os.path.exists(filepath):
            try:
                X, y = load_dataset(filepath)
                datasets[ds_id] = (X, y)
                print(f"Loaded {ds_id}: {X.shape[0]} samples, {X.shape[1]} features")
            except Exception as e:
                print(f"Failed to load {ds_id}: {e}")
        else:
            print(f"File not found: {filepath}")
    
    return datasets


if __name__ == "__main__":
    print("Dataset Utilities for MOBGA-AOS")
    print("================================\n")
    
    # Print expected dataset information
    print("Expected Dataset Specifications:")
    print("-" * 70)
    print(f"{'ID':<8} {'Name':<30} {'Features':<10} {'Classes':<10} {'Instances':<10}")
    print("-" * 70)
    for ds_id, info in DATASET_INFO.items():
        print(f"{ds_id:<8} {info['name']:<30} {info['n_features']:<10} {info['n_classes']:<10} {info['n_instances']:<10}")
    
    print("\n\nAttempting to load datasets...")
    
    # Try to load datasets
    datasets = prepare_all_datasets()
    
    if datasets:
        print(f"\nSuccessfully loaded {len(datasets)} datasets")
        
        # Print detailed summary for each
        for ds_id, (X, y) in datasets.items():
            print_dataset_summary(f"{ds_id}.csv")
    else:
        print("\nNo datasets found. Please ensure Datasets.zip is in the current directory.")
        print("Or manually place DS02.csv, DS04.csv, DS05.csv, DS07.csv, DS08.csv, DS10.csv")
