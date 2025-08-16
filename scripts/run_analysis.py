#!/usr/bin/env python3
"""
Example script to run GTZAN analysis
"""

import sys
sys.path.append('..')

from gtzan_analysis import GTZANAnalyzer

def main():
    """Run the complete GTZAN analysis pipeline."""
    
    # Check if data file exists
    import os
    if not os.path.exists('features_30_sec.csv'):
        print("❌ Error: features_30_sec.csv not found!")
        print("Please download the GTZAN dataset and place features_30_sec.csv in the root directory")
        print("Download from: http://marsyas.info/downloads/datasets.html")
        return
    
    print("🎵 Starting GTZAN Music Genre Analysis...")
    
    # Initialize analyzer
    analyzer = GTZANAnalyzer(
        data_path='features_30_sec.csv',
        output_dir='output'
    )
    
    try:
        # Run complete analysis
        analyzer.run_complete_analysis()
        print("✅ Analysis completed successfully!")
        print("📊 Check the 'output' directory for results")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return

if __name__ == "__main__":
    main()
