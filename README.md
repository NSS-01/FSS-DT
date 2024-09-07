# FSS-DT

## Introduction
The code of  FSS-DT is on [application/Lightweiht_disicion_tree](application/Lightweiht_disicion_tree).

This project is a secure multi-party computation library that designs and implements privacy-preserving computation protocols based on arithmetic secret sharing and function secret sharing.
It also utilizes these protocols to implement the application of privacy-preserving machine learning. 

## Installation tutorial
This project requires PyTorch>=1.8.0, and it is recommended to use PyTorch==1.8.0.
Other dependencies are listed in the ./requirements.txt file.
You can install the project dependencies by executing the following command:
```bash
pip install -r requirements.txt
```

## Getting start
All test codes needs to be run in the project root directory, as shown in the following example:
```
# Open two terminals and input the following code in each terminal:
python debug/application/neural_network/2pc/neural_network_server.py
python debug/application/neural_network/2pc/neural_network_client.py
```
If you cannot start using the above command, try adding the following code to the beginning of the corresponding test file:
```python
import sys
sys.path.append('/path/to/the/directory/mpctensorlib')
print(sys.path)
```

For instructions on how to use the library for a privacy application, please refer to the tutorials in the pack 'tutorials', which are presented as a Jupyter notebook, so please install the following in your conda environment:
```bash
conda install ipython jupyter
```
1.Tutorial_0_Before_Starting.ipynb - Before starting the tutorial, this notebook provides an introduction to the configuration information and auxiliary parameters required for computations in the library.  
2.Tutorial_1_Ring_Tensor.ipynb - This tutorial introduces the basic data type `RingTensor` in the library. It demonstrates how to perform basic operations using RingTensor.  
3.Tutorial_2_Arithmetic_Secret_Sharing.ipynb - This tutorial explains the basic data type, `ArithmeticSharedRingTensor`, used for secure multi-party computation in the library. It shows how to perform basic operations using ArithmeticSharedRingTensor through arithmetic secret sharing techniques that distribute data into two shares for two participating parties.  

## Architecture
- [application](application/Lightweiht_disicion_tree)  
The application package contains applications implemented using the functionalities of MPCTensorLib. Currently, it supports automatic conversion of plaintext cipher models and privacy-preserving neural network inference.
- [common](common)   
The common package includes general utilities and the basic data structures used by this lib, such as network communication, random number generators, and other tools.
- [config](config)   
The config package includes the basic configuration and network configuration of MPCTensorLib.
- [crypto](crypto)  The crypto package is the core of the lib and includes the privacy computation primitives and protocols.
- [tutorials](/)  
The tutorials package contains the usage tutorials of MPCTensorLib.
## License
MPCTensorLib is based on the MIT license, as described in LICENSE.
