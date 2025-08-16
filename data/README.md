# Data Directory

## GTZAN Dataset

This directory should contain the GTZAN music genre dataset.

### Required Files

- `features_30_sec.csv` - Pre-extracted audio features for 1000 tracks

### Download Instructions

1. Visit: http://marsyas.info/downloads/datasets.html
2. Download the GTZAN dataset
3. Extract `features_30_sec.csv` 
4. Place it in the `data/raw/` directory

### Dataset Description

- **1000 audio tracks** (30 seconds each)
- **10 genres**: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock
- **100 tracks per genre**
- **58 audio features** extracted using Librosa

### File Structure

```
data/
├── raw/
│   └── features_30_sec.csv     # Original dataset
└── processed/
    └── (processed data files)  # Generated during analysis
```
