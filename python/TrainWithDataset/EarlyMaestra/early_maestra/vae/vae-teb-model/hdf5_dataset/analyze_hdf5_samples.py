#!/usr/bin/env python3
"""
HDF5 Dataset Sample Counter and Analyzer

This script analyzes HDF5 datasets created by the create_hdf5_dataset.py script
and provides detailed information about sample counts and dataset structure.
"""

import h5py
import numpy as np
import os
from typing import Dict, List, Tuple
import argparse


def analyze_single_hdf5(file_path: str) -> Dict:
    """
    Analyze a single HDF5 file and return comprehensive statistics.
    
    Args:
        file_path: Path to the HDF5 file
        
    Returns:
        Dictionary containing analysis results
    """
    if not os.path.exists(file_path):
        return {"error": f"File not found: {file_path}"}
    
    try:
        with h5py.File(file_path, 'r') as f:
            # Basic information
            info = {
                "file_path": file_path,
                "file_size_mb": os.path.getsize(file_path) / (1024 * 1024),
                "datasets": {},
                "total_samples": 0,
                "sample_breakdown": {}
            }
            
            # Analyze each dataset
            for key in f.keys():
                dataset = f[key]
                dataset_info = {
                    "shape": dataset.shape,
                    "dtype": str(dataset.dtype),
                    "size_mb": dataset.nbytes / (1024 * 1024)
                }
                info["datasets"][key] = dataset_info
            
            # Get total sample count (assuming first dimension is samples)
            if len(f.keys()) > 0:
                first_key = list(f.keys())[0]
                info["total_samples"] = f[first_key].shape[0]
                
                # Analyze sample metadata if available
                if info["total_samples"] > 0:
                    # CS/BG label breakdown
                    if 'cs_label' in f and 'bg_label' in f:
                        cs_labels = f['cs_label'][()]
                        bg_labels = f['bg_label'][()]
                        
                        cs_true = np.sum(cs_labels == 1)
                        cs_false = np.sum(cs_labels == 0)
                        bg_true = np.sum(bg_labels == 1)
                        bg_false = np.sum(bg_labels == 0)
                        
                        info["sample_breakdown"] = {
                            "cs_label_true": int(cs_true),
                            "cs_label_false": int(cs_false),
                            "bg_label_true": int(bg_true),
                            "bg_label_false": int(bg_false)
                        }
                    
                    # GUID uniqueness check
                    if 'guid' in f:
                        guids = [g.decode('utf-8') if isinstance(g, bytes) else str(g) 
                                for g in f['guid'][()]]
                        unique_guids = len(set(guids))
                        info["unique_guids"] = unique_guids
                        info["guid_duplicates"] = info["total_samples"] - unique_guids
                    
                    # Epoch range
                    if 'epoch' in f:
                        epochs = f['epoch'][()]
                        info["epoch_range"] = {
                            "min": float(np.min(epochs)),
                            "max": float(np.max(epochs)),
                            "mean": float(np.mean(epochs)),
                            "std": float(np.std(epochs))
                        }
            
            return info
            
    except Exception as e:
        return {"error": f"Error reading {file_path}: {str(e)}"}


def analyze_dataset_directory(directory_path: str) -> Dict:
    """
    Analyze all HDF5 files in a directory and subdirectories.
    
    Args:
        directory_path: Path to directory containing HDF5 files
        
    Returns:
        Dictionary with analysis results for all files
    """
    results = {
        "directory": directory_path,
        "files_analyzed": 0,
        "total_samples_all_files": 0,
        "total_size_mb": 0,
        "file_results": {},
        "summary": {}
    }
    
    # Find all HDF5 files
    hdf5_files = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.hdf5') or file.endswith('.h5'):
                hdf5_files.append(os.path.join(root, file))
    
    print(f"Found {len(hdf5_files)} HDF5 files in {directory_path}")
    
    # Analyze each file
    for file_path in hdf5_files:
        print(f"Analyzing: {file_path}")
        file_result = analyze_single_hdf5(file_path)
        
        if "error" not in file_result:
            results["files_analyzed"] += 1
            results["total_samples_all_files"] += file_result["total_samples"]
            results["total_size_mb"] += file_result["file_size_mb"]
        
        # Use relative path as key for cleaner output
        rel_path = os.path.relpath(file_path, directory_path)
        results["file_results"][rel_path] = file_result
    
    # Generate summary statistics
    if results["files_analyzed"] > 0:
        sample_counts = [r["total_samples"] for r in results["file_results"].values() 
                        if "error" not in r]
        
        results["summary"] = {
            "files_with_data": len([c for c in sample_counts if c > 0]),
            "files_empty": len([c for c in sample_counts if c == 0]),
            "avg_samples_per_file": np.mean(sample_counts) if sample_counts else 0,
            "min_samples_per_file": np.min(sample_counts) if sample_counts else 0,
            "max_samples_per_file": np.max(sample_counts) if sample_counts else 0,
            "std_samples_per_file": np.std(sample_counts) if sample_counts else 0
        }
    
    return results


def print_analysis_results(results: Dict, detailed: bool = True):
    """Print analysis results in a formatted way."""
    
    if "directory" in results:
        # Directory analysis results
        print("="*80)
        print(f"HDF5 DATASET DIRECTORY ANALYSIS: {results['directory']}")
        print("="*80)
        print(f"Files analyzed: {results['files_analyzed']}")
        print(f"Total samples across all files: {results['total_samples_all_files']:,}")
        print(f"Total size: {results['total_size_mb']:.2f} MB")
        
        if results['summary']:
            print(f"\nSUMMARY STATISTICS:")
            print(f"  Files with data: {results['summary']['files_with_data']}")
            print(f"  Empty files: {results['summary']['files_empty']}")
            print(f"  Average samples per file: {results['summary']['avg_samples_per_file']:.1f}")
            print(f"  Min samples per file: {results['summary']['min_samples_per_file']:,}")
            print(f"  Max samples per file: {results['summary']['max_samples_per_file']:,}")
            print(f"  Std samples per file: {results['summary']['std_samples_per_file']:.1f}")
        
        print(f"\nPER-FILE BREAKDOWN:")
        print("-" * 80)
        
        for file_path, file_result in results['file_results'].items():
            if "error" in file_result:
                print(f"ERROR - {file_path}: {file_result['error']}")
            else:
                samples = file_result['total_samples']
                size_mb = file_result['file_size_mb']
                print(f"{file_path}: {samples:,} samples ({size_mb:.1f} MB)")
                
                if detailed and file_result['sample_breakdown']:
                    breakdown = file_result['sample_breakdown']
                    print(f"  CS labels - True: {breakdown.get('cs_label_true', 0)}, "
                          f"False: {breakdown.get('cs_label_false', 0)}")
                    print(f"  BG labels - True: {breakdown.get('bg_label_true', 0)}, "
                          f"False: {breakdown.get('bg_label_false', 0)}")
                
                if detailed and 'unique_guids' in file_result:
                    print(f"  Unique GUIDs: {file_result['unique_guids']}")
                    if file_result['guid_duplicates'] > 0:
                        print(f"  GUID duplicates: {file_result['guid_duplicates']}")
    
    else:
        # Single file analysis results
        if "error" in results:
            print(f"ERROR: {results['error']}")
            return
            
        print("="*80)
        print(f"HDF5 DATASET ANALYSIS: {results['file_path']}")
        print("="*80)
        print(f"Total samples: {results['total_samples']:,}")
        print(f"File size: {results['file_size_mb']:.2f} MB")
        
        print(f"\nDATASET STRUCTURE:")
        for name, info in results['datasets'].items():
            print(f"  {name}: {info['shape']} ({info['dtype']}) - {info['size_mb']:.1f} MB")
        
        if results['sample_breakdown']:
            print(f"\nSAMPLE BREAKDOWN:")
            breakdown = results['sample_breakdown']
            print(f"  CS labels - True: {breakdown.get('cs_label_true', 0)}, "
                  f"False: {breakdown.get('cs_label_false', 0)}")
            print(f"  BG labels - True: {breakdown.get('bg_label_true', 0)}, "
                  f"False: {breakdown.get('bg_label_false', 0)}")
        
        if 'unique_guids' in results:
            print(f"\nGUID ANALYSIS:")
            print(f"  Unique GUIDs: {results['unique_guids']}")
            if results['guid_duplicates'] > 0:
                print(f"  Duplicate GUIDs: {results['guid_duplicates']}")
        
        if 'epoch_range' in results:
            print(f"\nEPOCH STATISTICS:")
            epoch = results['epoch_range']
            print(f"  Range: {epoch['min']:.1f} to {epoch['max']:.1f}")
            print(f"  Mean: {epoch['mean']:.1f} ± {epoch['std']:.1f}")


def main():
    parser = argparse.ArgumentParser(description='Analyze HDF5 datasets for sample counts and structure')
    parser.add_argument('path', help='Path to HDF5 file or directory containing HDF5 files')
    parser.add_argument('--brief', action='store_true', help='Show brief output without detailed breakdowns')
    
    args = parser.parse_args()
    
    if os.path.isfile(args.path):
        # Analyze single file
        results = analyze_single_hdf5(args.path)
        print_analysis_results(results, detailed=not args.brief)
    elif os.path.isdir(args.path):
        # Analyze directory
        results = analyze_dataset_directory(args.path)
        print_analysis_results(results, detailed=not args.brief)
    else:
        print(f"Error: Path '{args.path}' is not a valid file or directory")
        return 1
    
    return 0


if __name__ == "__main__":
    # Example usage if run directly
    import sys
    
    if len(sys.argv) > 1:
        sys.exit(main())
    else:
        # Default behavior - analyze current directory
        print("No path specified. Analyzing current directory for HDF5 files...")
        # current_dir = "."
        # results = analyze_dataset_directory(current_dir)
        # print_analysis_results(results, detailed=True)
        
        # Also check for the specific file mentioned in the original code
        test_file = r"C:\Users\mahdi\Desktop\teb_vae_model\hdf5_dataset\train_dataset_cs.hdf5"
        if os.path.exists(test_file):
            print(f"\n{'='*80}")
            print("ANALYZING SPECIFIC FILE FROM PROJECT:")
            file_results = analyze_single_hdf5(test_file)
            print_analysis_results(file_results, detailed=True)