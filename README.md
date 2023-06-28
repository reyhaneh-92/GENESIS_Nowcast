# GENESIS_Nowcast
This repository contains the code for the Global prEcipitation Nowcasting using intEgrated multi-Satellite retrIevalS for GPM (GENESIS) nowcasting machine, as described in the paper titled "Global Precipitation Nowcasting of Integrated Multi-satellite Retrievals for GPM: A U-Net Convolutional LSTM Architecture."

The "Models" folder provides details on the proposed precipitation nowcasting neural networks, including the trained models. The remaining files are used to properly read the data, create, train, and save the models. Finally, the trained neural networks are utilized to predict precipitation on a global scale.

The "GENESIS notebook" demonstrates the procedure for using the trained neural networks to perform global precipitation nowcasting on a single sample.

The figure below shows the architecture of the developed network.


![244203125-52691118-6bbb-4dbd-bae0-ec73f3b0aef4](https://github.com/reyhaneh-92/GENESIS_Nowcast/assets/80214308/86c6595d-6886-476c-b265-20b10f72c938)

The video below shows the global prediction of GENESEIS algorithm for over five days.

https://github.com/reyhaneh-92/GENESIS_Nowcast/assets/80214308/ca5e0044-be8a-4e58-b3e1-e49165e5229d

