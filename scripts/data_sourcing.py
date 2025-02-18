#This function is to use kaggle and source the data. The code was found in the
#Kaggle documentation
import kagglehub
import os

def download_data():
    # Download latest version
    path = kagglehub.dataset_download("shuvokumarbasak4004/banana-tree-disease-detection-new-and-update-dataset")

    print("Path to dataset files:", path)

    #We path join an extra folder just to ensure we have all four classes for model
    #training
    corr_path = os.path.join(path, 'banana_tree')

    return corr_path
