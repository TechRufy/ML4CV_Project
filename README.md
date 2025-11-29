#  Overview

The goal of this work is to understand how semantic segmentation models behave when encountering unknown or out of distribution (OOD) objects in driving scenes.

The notebook includes:

- Dataset extraction and preprocessing  
- Full implementation of three anomaly detection techniques  
- Metric computation (AUPR, mIoU)  
- Visualizations of segmentation and anomaly maps  
- Comparison of the three approaches  

## ML4CV---Assignment

The notebook contains the assignment for the course of Machine Learning for Computer Vision at the University of Bologna.
The notebook has been created using Kaggle. All the model training and evaluation have been done with the HPC of the university of Bologna, in particular the nodes with an nvidia l40.

## Dataset and Data
Kaggle is fully cloud-based. Your local machine and the Kaggle server don’t share a file system, this implies that all the files need to be dinamically downloaded during the notebook execution, or manually added. To avoid this, i proceded as follows:
- The StreetHazards dataset has been directly downloaded from the Berkley university link.
- The weight of the various model are all automatically downloaded from the google drive, if you want to check them, you can find them here [weights](https://drive.google.com/file/d/1JjzTNzLbhqnWsASxSG3_kqpV_VjFWOPJ/view?usp=sharing)
- The external piece of code used in the notebook are automatically downloaded again from google drive, here you can check them [models](https://drive.google.com/drive/folders/16UjFV-zWcWO2mJPSbTBR4K0talyauq1d?usp=sharing)
- The model used for the dual decoder us pretrained weight that you can find here [pretrained](https://drive.google.com/drive/folders/11UBNiSbg1dvmvUCTnScMf0bb1P5QHnrl?usp=sharing)

##  Notebook Structure

1. Environment Setup
   - Install dependencies  
   - Download pretrained weights  

2. Utilities
   - Path and directory management  
   - Metrics (AUPR, IoU)  
   - Visualization helpers  

3. Dataset Loading
   - StreetHazards dataset  
   - Anomaly injected dataset
   - Creation of dataloaders  

4. Methods
   - Custom dual head segmentation model  
   - Standardized maximum logit  
   - DeepLabV3+ with FRE  

5. Evaluation
   - AUPR/mIoU computation  
   - Qualitative examples  
   - Final comparison
  
## Run the notebook
To ensure reproducibility please upload the notebook on Kaggle, connect to GPU P100 and run the notebook. Please, ensure that on your Kaggle notebook internet connection is enabled. 
It should automatically pin the environment to the original ones, do not change this otherwise some packages may give some errors (Kaggle environment changes often). Everything needed to run it is automatically downloaded.
