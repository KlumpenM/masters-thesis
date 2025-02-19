# Description
This is a project for a masters thesis at Aarhus University, focussing on Cross-Silo federated learning, Differential Privacy and Secure Multiparty Communicatino

# Dependency
Since we are only working with the flower framework (At the time of writing this) these are the depedencies we are using:
- Python (version 3.12)
- Torch
- Torchvision
- flwr
- matplotlib
- pfl[torch]

*IMPORTANT*: In order to install `pfl`, the required Python version is 3.10.

You might run into issues with importing `matplotlib.pyplot` due to following error `ImportError: cannot import name '_imaging' from 'PIL'`. To solve this, enter the following in the shell:
```
pip uninstall PIL
pip uninstall Pillow
pip install Pillow
```
Source: https://stackoverflow.com/questions/64998199/cannot-import-name-imaging-from-pil