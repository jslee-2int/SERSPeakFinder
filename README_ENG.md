# SERSPeakFinder

SERS Raman Spectrum Peak Detection — User Manual

## Table of Contents
1. [Program Overview](#program-overview)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [Main Features](#main-features)
5. [Program Execution](#program-execution)
6. [Detailed User Guide](#detailed-user-guide)
7. [Troubleshooting Guide](#troubleshooting-guide)
8. [Frequently Asked Questions](#frequently-asked-questions)

## Program Overview
This program is a comprehensive tool for processing and analyzing Raman spectroscopy data. It efficiently processes data obtained from Raman spectroscopy, automatically detects peaks using machine learning technology, and provides visualization and analysis capabilities.

### Key Features
- Intuitive Graphical User Interface (GUI)
- Deep Learning-based Automatic Peak Detection
- Real-time Data Visualization
- Batch Processing Support
- Automatic Report Generation

## System Requirements

### Hardware Requirements
- CPU: Dual-core or higher recommended
- RAM: 8GB or higher recommended
- Storage: Minimum 1GB free space
- GPU: NVIDIA GPU recommended (for deep learning model training)

### Software Requirements
- Operating System: Windows 10 or higher
- Python 3.8 or higher
- Required Libraries:
  - PyQt6: GUI framework
  - NumPy: Numerical computation
  - Pandas: Data processing
  - PyTorch: Deep learning framework
  - SciPy: Scientific computation
  - Matplotlib: Data visualization

## Installation

1. Install Python
   ```bash
   # Install Python 3.8 or higher
   ```

2. Create and Activate Virtual Environment
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. Install Required Libraries
   ```bash
   pip install PyQt6 numpy pandas torch scipy matplotlib
   ```

## Main Features

### 1. Data Preprocessing
- Original Data Format Conversion
- Noise Reduction
- Baseline Correction
- Data Normalization

### 2. Peak Detection
- CNN-based Automatic Peak Detection
- Manual Peak Detection and Verification
- Batch Processing Support

### 3. Data Analysis
- Spectrum Visualization
- Statistical Analysis
- Result Comparison and Verification

### 4. Report Generation
- CSV Format Result Storage
- Detailed Analysis Report Generation
- Automatic Graph Generation

## Program Execution

1. Start Program
   ```bash
   python main.py
   ```

2. Initial Setup
   - Check config.ini file
   - Modify settings if necessary

## Detailed User Guide

### 1. Data Conversion Tab
#### Feature Overview
- Convert original Raman spectrum data into analyzable format
- Separate coordinate-based data into individual spectrum files

#### Detailed Instructions
1. Select Data File
   - Click 'Browse' button
   - Select original data file (.txt or .csv format)
   - Supported format: Tab-delimited text file

2. Output Settings
   - Select output folder
   - Files are automatically saved in (x_y).csv format

3. Execute Conversion
   - Click 'Process' button
   - Monitor conversion progress
   - File list automatically updates upon completion

4. Check Results
   - Select converted files from left list
   - View spectrum in right graph area
   - Use mouse drag to zoom in/out

### 2. Data Classification Tab
#### Feature Overview
- Manually classify peak presence in spectrum data
- Label data for training dataset creation

#### Detailed Instructions
1. Data Folder Setup
   - Click 'Browse' button to select folder with CSV files
   - File list automatically loads in treeview

2. Peak Detection Range Setup
   - Center Wavelength: Center wavelength of region of interest
   - Range: Search range from center wavelength
   - Vertical lines display on graph after input

3. Data Classification Process
   - Spectrum automatically displays upon file selection
   - Quick classification using keyboard shortcuts:
     - 'A': Peak present (1)
     - 'D': No peak (0)
   - Use arrow keys to move between files

4. Save Classification Results
   - Automatically saves as .dat file
   - Use 'Run Classifier' button to save complete results as CSV

### 3. Model Training Tab
#### Feature Overview
- Train CNN model for peak detection
- Real-time training progress monitoring

#### Detailed Instructions
1. Data Setup
   - Training Data Folder: Select training data folder
   - Info File: Select CSV file with label information
   - Model Save Path: Specify path for trained model

2. Hyperparameter Setup
   - Epochs: Number of training iterations (default: 300)
   - Batch Size: Processing batch size (default: 64)
   - Test Size: Test data ratio (default: 0.2)
   - Random State: Random seed value
   - Validation Split: Validation data ratio (default: 0.2)

3. Execute Training
   - Click 'Start Training' button
   - Real-time progress display:
     - Current Epoch
     - Train/Test Loss
     - Train/Test Accuracy
   - Real-time graph updates:
     - Loss graph
     - Accuracy graph

4. Check Results
   - Best performance model automatically saved
   - Training log available for review

### 4. Peak Detection Tab
#### Feature Overview
- Automatic peak detection using trained model
- Batch processing support

#### Detailed Instructions
1. Model Setup
   - Model File: Select trained model file
   - Target Folder: Select data folder for analysis
   - Batch Size: Set processing batch size

2. Execute Detection
   - Click 'Start Detection' button
   - Monitor progress through progress bar
   - Results table updates automatically

3. Check Results
   - File name
   - Peak detection result (0 or 1)
   - Confidence score
   - Use ↑↓ keys to browse results and view graphs

4. Save Results
   - Click 'Save to CSV' button to save results
   - Includes filename, detection results, confidence scores

### 5. Result Comparison Tab
#### Feature Overview
- Compare CNN model detection results with manual classification
- Accuracy analysis and statistical information

#### Detailed Instructions
1. Select Comparison Files
   - CNN Model Result: Model detection result file
   - Manual Classification: Manual classification result file

2. Execute Comparison
   - Click 'Match Result' button
   - Comparison results calculated automatically

3. Result Analysis
   - Match/Mismatch count
   - Overall accuracy
   - Misclassification detailed analysis:
     - Peak present misclassification count
     - No peak misclassification count

4. Save Results
   - Click 'Save Report' button for detailed report
   - Includes file-by-file comparison results

### 6. Integration Analysis Tab
#### Feature Overview
- Quantitative analysis of spectrum data
- Savitzky-Golay filtering and integration calculation

#### Detailed Instructions
1. Data Setup
   - Report File: Select report file for analysis
   - Data Folder: Select spectrum data folder

2. Analysis Parameter Setup
   - Window Length: SG filter window size
   - Polyorder: Polynomial order
   - Center Wavelength: Analysis center wavelength
   - Range Value: Analysis range

3. Execute Analysis
   - Click 'Process' button
   - Results table automatically generated:
     - File name
     - Peak presence
     - Integration value

4. Result Visualization
   - Original data
   - Filtered data
   - Integration area display
   - Use ↑↓ keys to browse results

## Troubleshooting Guide

### Common Issues
1. Program Won't Start
   - Check Python version
   - Verify required libraries installation
   - Check virtual environment activation

2. Data Loading Errors
   - Check file format
   - Check for special characters in file path
   - Verify file access permissions

3. Model Training Issues
   - GPU memory shortage: Reduce batch size
   - Overfitting: Adjust validation set ratio
   - Unstable training: Adjust learning rate

### Performance Optimization
1. Data Processing Speed Improvement
   - Optimize batch size
   - Clean unnecessary files
   - Periodically delete temporary files

2. Memory Usage Management
   - Use batch processing for large files
   - Run garbage collection periodically
   - Release unnecessary data immediately

## Frequently Asked Questions

### Data Related
Q: What input file formats are supported?
A: Tab-delimited text files (.txt) or CSV files (.csv) are supported.

Q: Can I change the file naming format during data conversion?
A: Currently fixed to (x_y).csv format.

### Model Training Related
Q: What are the optimal hyperparameters?
A: While dependent on dataset characteristics, generally recommended values are:
- Epochs: 300
- Batch Size: 64
- Test Size: 0.2
- Validation Split: 0.2

Q: Can I use trained models with different datasets?
A: Yes, but new dataset characteristics should be similar to training data.

### Result Analysis Related
Q: How do I interpret peak detection confidence scores?
A: Values range from 0 to 1, with values closer to 1 indicating higher probability of peak presence.

Q: What are the units for integration analysis results?
A: Relative units, product of spectrum intensity and wavelength range.

## Settings Storage
- All settings automatically saved in `config.ini` file
- Auto-save on program exit
- Manual save: Use 'Save Config' button
- Saved settings include:
  - File paths
  - Analysis parameters
  - Model settings
  - Other user settings 