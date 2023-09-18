# Static voltage stability AI

Over the upcoming weeks, this webpage will receive new content. Following the successful completion of the Master's thesis defence, a revised edition of the thesis will be published here.

## Install
Please use Python version 3.9. Use pythonpackages_install.py to install the Python packages. This will ensure that all packages are compatible with the installed variants.

## Load Flow calculation / static voltage stability evaluation
There are 3 variants that are presented. Load flow calculation with PowerFactory, Netwon-Raphson and Gaus-Seidel. The programming was designed in such a way that the variants are in extra files that can be called up. As an example, code_comparison.py is shared. Multiprocessing was used.

The data generation is shown in main_powerfactory.py. It always generates 1000 cases. Since the memory prevents a fast calculation at higher runtimes, the program is called again by executer.py each time and the memory is cleared by the new call.
