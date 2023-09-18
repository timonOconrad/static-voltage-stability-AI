# Static voltage stability AI

This page will be updated in the coming weeks. After the successful defence of the Master's thesis, a revised edition of the thesis will be published here.

## Installation
Please use Python version 3.9. Use pythonpackages_install.py to install the Python packages. This will ensure that all packages are compatible with the installed variants.

You will also need to install Nividea CuDa (GPU interface) and CuDNN (Deep Neural Network). Follow the instructions that are specific to the graphics card in use.

## Load Flow Calculation / Static Stress Analysis
3 variants are presented. Load flow calculation using PowerFactory, Netwon-Raphson and Gaus-Seidel. The programming has been designed so that the variants are in extra files that can be called. As an example, code_comparison.py is provided. Multiprocessing was used.

The data generation is shown in main_powerfactory.py. It always generates 1000 cases. Since the memory prevents a fast computation at higher runtimes, the program is called again by executer.py each time and the memory is cleared by the new call.

The generated dataset with 789000 cases can be downloaded via this link, as GitHub only allows 25 MB without Git Large File Storage: https://www.dropbox.com/scl/fi/omuifzdu60k9sl3vl9lr3/updated_parquet_file.parquet?rlkey=yws01l5duxdkwjckreay0oqqb&dl=0

## AI
For the AI, the variant Vcon10Fc-2 was uploaded as an example. In the feature reduction only columns were removed and in a different architecture of the AI the number of neurons and the activation function were changed. 

Larger datasets were used for scaling. These are shared as .csv.

## Tools
Tool to extract the Y_Matrix from powerfactory. Currently without transformers.
