# Automated ROI Placement in MRI-PDFF Phantoms: A Deep Learning Approach to Quality Control

## Background
Quality control phantoms are routinely included in clinical and research MRI-PDFF liver protocols to ensure accurate fat quantification. Currently, analyzing phantom data requires manual placement of regions of interest (ROIs), a process that is both time-consuming and prone to human error. This project aims to automate ROI placement in MRI-PDFF phantoms using a deep learning approach to improve efficiency and consistency.

## Dataset
This study involved a retrospective analysis of liver MRI-PDFF exams acquired from 30 patients. Imaging was performed using a combination of IDEAL-IQ, Iron Quant, and flip-angle modulated (FAM) sequences. Each exam included a quality-control phantom consisting of five cylindrical compartments filled with gel materials of known fat fractions, spanning a range of 0–40% proton density fat fraction (PDFF). The phantom used was commercially available from Calimetrix (Madison, WI) and was included in the field of view for all scans. These known PDFF values served as ground truth references to support model training and evaluation for automated region-of-interest (ROI) segmentation and fat quantification.

## Model Architecture
We present a deep learning model based on a U-Net encoder-decoder architecture was developed to automate region-of-interest (ROI) segmentation in liver MRI-PDFF quality-control phantoms. The model was designed to classify voxels into seven categories, corresponding to the five cylindrical compartments of the phantom, background, and imaging artifacts. The network consisted of six encoding-decoding layers with filter sizes of [32, 64, 128, 256, 512, 1024], incorporating batch normalization, max pooling, and dropout regularization (increasing from 0.1 to 0.3 across layers). Training was performed using Dice loss to optimize segmentation accuracy, with the Adam optimizer and a softmax activation function in the final layer. Model performance was evaluated by comparing automated and manual ROI-based PDFF measurements using Bland-Altman analysis to assess agreement and quantify bias and limits of agreement.

## Running the Code
This project was developed and tested in a TensorFlow 2.5 environment. To reproduce results or run the pipeline:

### Environment Setup

Activate the required environment using Anaconda:

```bash
conda activate tf2.5
```
### Folder and File Descriptions

The repository is organized as follows:

- **`cylinder.py`**  
  Core script containing the U-Net model architecture and training pipeline.  
  - Trains the model on MRI-PDFF phantom data  
  - Saves Dice scores and loss values  
  - Outputs sample predictions during training

- **`thoraxProcessing.ipynb`**  
  Notebook for generating segmentation predictions from the trained model.  
  - Loads the model  
  - Applies segmentation to new MRI-PDFF scans  
  - Extracts automated ROIs for each of the five phantom cylinders

- **`postProcessing2.ipynb`**  
  Notebook for post-processing and evaluation.  
  - Performs Bland-Altman analysis  
  - Generates agreement plots between manual and automated ROI measurements

- **`final_stats.xlsx`**  
  Results file summarizing PDFF measurements.  
  - Compares ground truth vs. model-predicted values across all patients  
  - Organized per cylinder for quantitative evaluation
