# PV-Facility-Detection
A paper replicating project.

Automatic Detection of Photovoltaic Facilities using the Enhanced U-NET method
https://doi.org/10.1117/1.JRS.17.014516

ALL UPDATED CODE IS PRESENT ONLY IN "PV_Facility_Detection.ipynb". This includes training the multi-spectral images.

## Steps to run
1. Download the code.
2. Install all the requirements.
3. Ensure the annotations and images are present in one folder. Enter the path of that folder in the hyperparameters section.
4. Run train.py.

## Steps Involved

### Dataset Preparation

1. Create an account on scihub. This will give access to various Sentinel Satellite image data.

[](http://scihub.copernicus.eu/dhus)

1. The required area is zoomed and scanned for obtaining various Sentinel - 2 S2A products. The appropriate images with minimal cloud cover are chosen and downloaded.
    - Each image is approximately 1 GB large and contains  geo-spatial and spectral image data.
2. The QGIS tool is used to add the TCI (True Color Image) layer of each image only. The required area that mainly contains the PV Facilities is extracted, which were saved as separate images in JPEG format.
    
    This was repeated for all downloaded S2A images.
    
3. RoboFlow was used to annotate the images. The software uses an AI enhanced smart polygon for manually annotating the several cropped images of the PV Facilities. 
4. The dataset is now preprocessed - separated into 2x2 tiles to increase the number of images followed by Data Augmentation - horizontal and vertical rotation. The total number of images available are 992.
5. The images are imported and downloaded, with the annotations contained in a JSON file.  

### The Model

A basic U-NET is implemented in PyTorch. The architecture is given below:

![image](https://github.com/DarkSamurai1907/PV-Facility-Detection/assets/99598258/b5436f5b-4df0-4273-aef8-4dbbf2bda17e)


### Data loading and other utils

- The data is split into a ratio of 70:15:15 for train, validation and test respectively.
- The annotations.json file is read and is converted into mask labels for every image.
- Batches of size 32 are created for each set and are loaded into Data Loaders.
- Functions are created for saving and loading checkpoints, creating the data loaders, validating and checking accuracy (after every epoch) and saving predictions as images.

### Hyper Parameters

- An initial learning rate of 1e-5 is set along with a learning rate scheduler which gradually decreases the learning rate by a factor of 0.1 if the validation loss starts plateauing, with a minimum learning rate of 1e-7. The model ceases training if the validation loss doesn’t improve for 5 epochs.
- An initial 50 number of epochs is set, but the training ceases before it reaches this value.

### Final Test Metrics

### Changes to the initial version

The second version will include high quality advanced images of 12 channels carrying spectral as well as spatial data
