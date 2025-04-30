<div align="center">

# **Peritumoral-Aware MIXture learning (PAMIX)**

<div align="center">

## 🔥🔥🔥

#### Updated on April 30 , 2025

</div>


<hr style=" height:2px;border:none;border-top:2px dotted #185598;" />

## ✨Paper

This repository provides the official implementation of **Multi-Level Peritumoral-Aware Mixture Learning for Breast Tumor Diagnosis** 




## ✨Dataset

|  Organ  |          Images          |     Categories     |                            Links                             |
| :-----: | :----------------------: | :----------------: | :----------------------------------------------------------: |
| BUS-BRA |           1875           | Malignant & Benign |        [BUS-BRA](https://zenodo.org/records/8231412)         |
|  BUSI   | 647 (exclude 133 normal) | Malignant & Benign | [BUSI](https://www.kaggle.com/datasets/subhajournal/busi-breast-ultrasound-images-dataset) |
| QAMEBI  |           232            | Malignant & Benign | [QAMEBI](https://qamebi.com/breast-ultrasound-images-database/) |



## ✨Model & Weights

|  Models   |                     Weights for BUS-BRA                      | Weights for BUSI | Weights for QAMEBI |
| :-------: | :----------------------------------------------------------: | :--------------: | :----------------: |
|   PAMIX   | [Weights](https://drive.google.com/file/d/1neCewcwssZMjPhGmzs2AubSr4g3B0IxP/view?usp=drive_link) |     Weights      |      Weights       |
| Uni-PAMIX | [Weights](https://drive.google.com/file/d/1wrmwtDUlOneW5N-ukyviJXcpSRhGhy8h/view?usp=drive_link) |     Weights      |      Weights       |



## ✨Installation & Preliminary

1. Clone the repository.
    ```
    git clone https://github.com/Git-HB-CHEN/PAMIX-Family.git
    cd PAMIX-Family
    ```
    
2. Create a virtual environment and activate the environment.
    ```
    conda create -n PAMIX-Family python=3.8
    conda activate PAMIX-Family
    ```
    
3. Install Pytorch == 1.13.0+cu117, and torchvision==0.14.0+cu117.
   (You can follow the instructions [here](https://pytorch.org/get-started/locally/))

4. Install other dependencies.
   ```
   pip install -r requirements.txt
   ```



## ✨Inference using PAMIX & Uni-PAMIX

1. Download the `Weights` of the PAMIX-Family

2. Place your images in the `examples` folder

3. Infer your images with the PAMIX
   ```python
   python infer_PAMIX.py
   ```

4. Infer your images with the Uni-PAMIX

   ```
   python infer_Uni-PAMIX.py
   ```

   

_***The relevant codes and weights are under preparation and will be made available soon.***_