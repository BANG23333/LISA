# LISA
LISA: Learning-Integrated Space Partitioning Framework for Traffic Accident Forecasting on Heterogeneous Spatiotemporal Data

# Abstract
Traffic accident forecasting is an important task for intelligent transportation management and emergency response systems. However, this problem is challenging due to the spatial heterogeneity of the environment. Existing data-driven methods mostly focus on studying homogeneous areas with limited size (e.g. a single urban area such as New York City) and fail to handle the heterogeneous accident patterns over space at different scales. Recent advances (e.g. spatial ensemble) utilize pre-defined space partitions and learn multiple models to improve prediction accuracy. However, external knowledge is required to define proper space partitions before training models and pre-defined partitions may not necessarily reduce the heterogeneity. To address this issue, we propose a novel Learning-Integrated Space Partition Framework (LISA) to simultaneously learn partitions while training models, where the partitioning process and learning process are integrated in a way that partitioning is guided explicitly by prediction accuracy rather than other factors. Experiments using real-world datasets, demonstrate that our work can capture underlying heterogeneous patterns in a self-guided way and substantially improve baseline networks by an average of 12.7%.

![alt text](https://github.com/BANG23333/LISA/blob/main/img/Multi-SP.png)

# Environment
- python 3.7.0
- torch 1.12.1
- matplotlib 3.5.2
- numpy 1.21.5
- sklearn 1.1.1

# Run LISA
Modify path and Run LisaNet.py

Data can be downloaded through [link](https://drive.google.com/file/d/11mFFlVU_pC0xj0yonrZ-bE3wauRI1rqx/view?usp=sharing)
