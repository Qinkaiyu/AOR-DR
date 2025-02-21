# AOR-DR
### Datasets 
Data can be obtained from [here](https://github.com/chehx/DGDR/blob/main/GDRBench/README.md).

Your dataset should be organized as: 

Adjust target_DATA(need change)_train according to different targets of your DG test.

```
.
├── images
│   ├── DATASET
│   │   ├── mild_npdr
│   │   ├── moderate_npdr
│   │   ├── nodr
│   │   ├── pdr
│   │   └── severe_npdr
│   ├── DATASET2
│   │   ├── mild_npdr
│   │   ├── moderate_npdr
│   │   ├── nodr
│   │   ├── pdr
│   │   └── severe_npdr
│   ├── DATASET3
│   │    ...
│   ...  ...
│  
│   
└── splits
    ├── target_DATA(need change)_train.txt
    ├── DATA(need change)_crossval.txt
    ...

```
### train
code will upload after accept.
### test
``` bash
python test.py
```
### checkpoint
[link](https://drive.google.com/drive/folders/1AFiUMh3WB53XOMwYVm-2_6ooOcram77l?usp=sharing)
