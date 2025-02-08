# A Deep Learning Framework and Medium-Scale Drone Dataset for Fire Detection in Complex Surveillance

# Project Structure
Please follow the project structure for successful implementation. 
```
project/
├── dataset/
│   ├── training/
│   ├── validation/
│   └── testing/
├── results/
│   ├── model/
│   ├── plots/
│   └── metrics/
├── main.py
├── config.py
├── data_handling.py
├── model_definition.py
├── training.py
├── evaluation.py
├── README.md
└── requirements.txt
```

## Dataset
Place your dataset in the dataset directory with the following structure:
```
dataset/
├── training/
│   ├── fire/
│   └── non_fire/
├── validation/
│   ├── fire/
│   └── non_fire/
└── testing/
    ├── fire/
    └── non_fire/
```
Usage
Run the main script to train and evaluate the model:

```python main.py```

The script will train the model using the training and validation datasets, and then evaluate it using the testing dataset. The results, including confusion matrix, ROC curve, and precision-recall curve, will be displayed.
## Results
The model's performance is evaluated using various metrics, including accuracy, confusion matrix, ROC curve, and precision-recall curve. The results are displayed as plots and printed in the console. Note please create a folder in the structure, as I have given above. 

## 1 Paper Link 
Paper will be available after publication.
## 2 Dataset
The datasets can be downloaded from the following links.

Option 1: The proposed DF dataset Download from given link: Click [here](https://github.com/SufyanDanish/DF-Dataset/edit/main/DF.html)

Option 2: Download FLAME’s dataset from the given link: Click [here](https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs)

Option 3:  Downlod FD dataset from the given link Click [here](https://drive.google.com/drive/folders/14wGLPGCoJCPwfJY0PeK9tha64MqUF9iG?usp=sharing))
## 3 Citation and Acknowledgements
