

# Introduction

This is a temporary repository dedicated to review, which will be moved to our team's main library after the review is completed.
# FSS-DT
FSS-DT is implemented based on our team's MPCSensorLib, and its source code is stored in [application/Lightweiht_disition_tree](https://github.com/NSS-01/FSS-DT/tree/main/application/Lightweiht_disicion_tree)

# MPCTensorLib
## Installation tutorial

This project requires PyTorch>\=1.8.0, and it is recommended to use PyTorch\=\=1.8.0. Other dependencies are listed in the ./requirements.txt file. You can install the project dependencies by executing the following command:

```bash
pip install -r requirements.txt
```

## Getting start

All test codes needs to be run in the project root directory, as shown in the following example:

    # Open two terminals and input the following code in each terminal:
    python debug/application/neural_network/2pc/neural_network_server.py
    python debug/application/neural_network/2pc/neural_network_client.py

If you cannot start using the above command, try adding the following code to the beginning of the corresponding test file:

```python
import sys
sys.path.append('/path/to/the/directory/mpctensorlib')
print(sys.path)
```
## Architecture

*   [application](https://github.com/NSS-01/FSS-DT/tree/main/application)\
    The application package contains applications implemented using the functionalities of MPCTensorLib. Currently, it supports automatic conversion of plaintext cipher models and privacy-preserving neural network inference.

*   [common](https://github.com/NSS-01/FSS-DT/tree/main/common)\
    The common package includes general utilities and the basic data structures used by this lib, such as network communication, random number generators, and other tools.

*   [config](https://github.com/NSS-01/FSS-DT/tree/main/config)\
    The config package includes the basic configuration and network configuration of MPCTensorLib.

*   [crypto](https://github.com/NSS-01/FSS-DT/tree/main/master/crypto)\
    The crypto package is the core of the lib and includes the privacy computation primitives and protocols.

*   [data](https://github.com/NSS-01/FSS-DT/tree/main/data)\
    The data package includes the auxiliary parameters used by the library for computation and the models and datasets used for privacy-preserving neural network inference.

*   [debug](https://github.com/NSS-01/FSS-DT/tree/main/debug)\
    The debug package includes the test code for the lib.

*   [model](https://github.com/NSS-01/FSS-DT/tree/main/model)\
    The model package includes system models and threat models used by the lib, such as the client-server model under semi-honest assumptions.

*   [tutorials](https://github.com/NSS-01/FSS-DT/tree/main/tutorials)\
    The tutorials package contains the usage tutorials of MPCTensorLib.

## License

MPCTensorLib is based on the MIT license, as described in LICENSE.
