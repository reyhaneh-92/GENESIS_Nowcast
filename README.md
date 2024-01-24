# GENESIS_Nowcast
This repository contains the code for the Global prEcipitation Nowcasting using intEgrated multi-Satellite retrIevalS for GPM (GENESIS) nowcasting machine, as described in the paper titled "Global Precipitation Nowcasting of Integrated Multi-satellite Retrievals for GPM: A U-Net Convolutional LSTM Architecture."

The "Models" folder provides details on the proposed precipitation nowcasting neural networks, including the trained models. The remaining files are used to properly read the data, create, train, and save the models. Finally, the trained neural networks are utilized to predict precipitation on a global scale.

The "GENESIS notebook" demonstrates the procedure for using the trained neural networks to perform global precipitation nowcasting on a single sample.

The figure below shows the architecture of the developed network.

![network](https://github.com/reyhaneh-92/GENESIS_Nowcast/assets/80214308/b805b353-700d-48f7-a43c-c14a9a91371b)
