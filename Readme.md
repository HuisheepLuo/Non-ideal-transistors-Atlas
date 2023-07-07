# Non-ideal transistors Atlas
Non-ideal transistor Atlas is a Python based tool for the identification of the non-ideal effect transistors.

The classification approach is a simple MLP network based on Pytorch. It could classify non-ideal effects of transistors, namely charge transport, charge injection, charge trapping, and dynamic ionic capacitance.

The attribution approach is a approach that helps researchers quantitatively evaluate and compare different device characteristics, facilitating better theoretical induction and experimental progress, using integrated-gradient method. The codes are modified based on the repository ["integrated-gradient-pytorch"](https://github.com/TianhongDai/integrated-gradient-pytorch).

The webui is a browser interface based on [Gradio](https://gradio.app) framework, and aims to use the tools easily. The matching process uses various algorithms to find the best-fit model for the experimental data. The webpage link: https://huggingface.co/spaces/shaken/Non-ideal-transistors-Atlas-v1

## Installation
To install Non-ideal model Atlas, simply clone this repository and install the required packages using pip:
```
pip install -r requirements.txt
```

## Usage
The `.py` files under the directory is the example files. Please run them in the installed python environment.

`1-create_dataset.py` is used for generating dataset.

`2-train_network.py` is used for training the MLP network by the dataset.

`3-attribution.py` is used for calculating the integrated gradients of the network and matching the curves.

`4-webui.py` is used for loading the web user interface.

## Related work
One of the non-ideal transistor applications are available in the repository ["Ion-transistor-network"](https://github.com/HuisheepLuo/Ion-transistor-network).