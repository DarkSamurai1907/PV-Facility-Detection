# PV-Facility-Detection

Automatic Detection of Photovoltaic Facilities using the Enhanced U-NET method.

A paper replicating project.

Zixuan Dui, Yongjian Huang, Jiuping Jin, Qianrong Gu, "Automatic detection of photovoltaic facilities from Sentinel-2 observations by the enhanced U-Net method," J. Appl. Rem. Sens. 17(1) 014516 (6 March 2023) https://doi.org/10.1117/1.JRS.17.014516

**ALL UPDATED CODE IS PRESENT IN "PV_Facility_Detection.ipynb"**. 
This includes training the multi-spectral images.

## Steps to run
1. Download the code.
2. Install all the requirements.
3. Ensure the annotations and images are present in one folder. Enter the path of that folder in the hyperparameters section.
4. Run train.py.

## Steps Involved

### Dataset Preparation

1. An account is created on []([http://scihub.copernicus.eu/dhus](https://dataspace.copernicus.eu/)). This will provide access to Sentinel satellite image data.

2. The required area is zoomed and scanned for obtaining various Sentinel - 2 S2A products. The appropriate images with minimal cloud cover are chosen and downloaded.
    - Each image is approximately 1 GB large and contains  geo-spatial and spectral image data.
3. The QGIS tool is used to add the TCI (True Color Image) layer of each image only. The required area that mainly contains the PV Facilities is extracted, which were saved as separate images in JPEG format.
    
    This was repeated for all downloaded S2A images.
    
4. RoboFlow was used to annotate the images. The software uses an AI enhanced smart polygon for manually annotating the several cropped images of the PV Facilities. 
5. The dataset is now preprocessed - separated into 2x2 tiles to increase the number of images followed by Data Augmentation - horizontal and vertical rotation. The total number of images are 992.
6. The images are imported and downloaded, with the annotations contained in a JSON file.  

### The Model

A basic U-NET is implemented in PyTorch. The enhanced U-Net is coded in the notebook file. It contains an additional U-Net called Multi-Spectral 3D convolution path and a Multi-Scale Pooling Block as enhancements to the original architecture.

### Data loading and other utils

- The data is split into a ratio of 70:15:15 for train, validation and test respectively.
- The annotations.json file is read and is converted into mask labels for every image.
- Batches of size 32 are created for each set and are loaded into Data Loaders.
- Functions are created for saving and loading checkpoints, creating the data loaders, validating and checking accuracy (after every epoch) and saving predictions as images.

### Hyper Parameters

- An initial learning rate of 1e-5 is set along with a learning rate scheduler which gradually decreases the learning rate by a factor of 0.1 if the validation loss starts plateauing, with a minimum learning rate of 1e-7. The model ceases training if the validation loss doesn’t improve for 5 epochs.
- An initial 50 number of epochs is set, but the training ceases before it reaches this value.
