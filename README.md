# GENESIS_Nowcast
This repository contains the code for the Global prEcipitation Nowcasting using intEgrated multi-Satellite retrIevalS for GPM (GENESIS) nowcasting machine, as described in the paper titeled "Global Precipitation Nowcasting of Integrated Multi-satellite Retrievals for GPM: A U-Net Convolutional LSTM Architecture."

The "Models" folder provides details on the proposed precipitation nowcasting neural netrowks, including the trained models. The remaining files are used to properly read the data, create, train, and save the models. Finally, the trained nueral networks are utilized to predict precipitation on a global scale.

The "GENESIS notebook" demonstrates the procedure for using the trained neural networks to perform global precipitation nowcasting on a single sample.

Figure below shows the atchitecture of the developed network.
<img src="[https://camo.githubusercontent.com/..." data-canonical-src="https://gyazo.com/eb5c5741b6a9a16c692170a41a49c858.png](https://github-production-user-asset-6210df.s3.amazonaws.com/46690843/244100993-086b40db-d02d-485c-a5f2-caa8793fb572.PNG?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20230607%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230607T202256Z&X-Amz-Expires=300&X-Amz-Signature=bfa8de89d9ff10f64b1930ae556ed3df477dc1e4c149860e56d4d6a0ac03e127&X-Amz-SignedHeaders=host&actor_id=80214308&key_id=0&repo_id=650361206)" width="200" height="400" />
