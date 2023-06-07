# GENESIS_Nowcast
This repository contains the code for the Global prEcipitation Nowcasting using intEgrated multi-Satellite retrIevalS for GPM (GENESIS) nowcasting machine, as described in the paper titeled "Global Precipitation Nowcasting of Integrated Multi-satellite Retrievals for GPM: A U-Net Convolutional LSTM Architecture."

The "Models" folder provides details on the proposed precipitation nowcasting neural netrowks, including the trained models. The remaining files are used to properly read the data, create, train, and save the models. Finally, the trained nueral networks are utilized to predict precipitation on a global scale.

The "GENESIS notebook" demonstrates the procedure for using the trained neural networks to perform global precipitation nowcasting on a single sample.

Figure below shows the atchitecture of the developed network.

![GENESIS_arch](https://github.com/reyhaneh-92/GENESIS_Nowcast/assets/46690843/52691118-6bbb-4dbd-bae0-ec73f3b0aef4)

