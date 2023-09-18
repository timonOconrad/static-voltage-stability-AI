# Static voltage stability AI

Over the upcoming weeks, this webpage will receive new content. Following the successful completion of the Master's thesis defence, a revised edition of the thesis will be published here.

## Install
Please use Python version 3.9. Use pythonpackages_install.py to install the Python packages. This will ensure that all packages are compatible with the installed variants.

## Load Flow calculation / static voltage stability evaluation
There are 3 variants that are presented. Load flow calculation with PowerFactory, Netwon-Raphson and Gaus-Seidel. The programming was designed in such a way that the variants are in extra files that can be called up. As an example, code_comparison.py is shared. Multiprocessing was used.

The data generation is shown in main_powerfactory.py. It always generates 1000 cases. Since the memory prevents a fast calculation at higher runtimes, the program is called again by executer.py each time and the memory is cleared by the new call.

The created Dataset with 789000 cases can be downloaded via this link ,as GitHub allowes only 25 MB without Git Large File Storage: https://www.dropbox.com/scl/fi/omuifzdu60k9sl3vl9lr3/updated_parquet_file.parquet?rlkey=yws01l5duxdkwjckreay0oqqb&dl=0

## KI
For the AI, variant Vcon10Fc-2 was uploaded as an example. In the reduction of featues, only columns were removed and in another architecture of the AI, the number of neurons and activation function were changed. 

Larger datasets were used for the scaling. These are shared as .csv.

## Tools
Tool for the extraction of the Y_Matrix from powerfactory. Currently without transformers.
