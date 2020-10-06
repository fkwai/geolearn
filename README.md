This code contains deep learning code used to modeling hydrologic systems, including soil moisture, streamflow and water quality, from projection to forecast. 

# Publications from this repo

K. Fang, M. Pan, CP. Shen, The Value of SMAP for Long-Term Soil Moisture Estimation With the Help of Deep Learning, IEEE Transactions on Geoscience and Remote Sensing (2018) https://doi.org/10.1109/TGRS.2018.2872131

K. Fang, CP. Shen, D. Kifer and X. Yang, Prolongation of SMAP to Spatio-temporally Seamless Coverage of Continental US Using a Deep Learning Neural Network, Geophysical Research Letters  (2017) https://doi.org/10.1002/2017GL075619

K. Fang, D. Kifer, K. Lawson, CP. Shen, Evaluating the potential and challenges of an uncertainty quantification method for long short-term memory models for soil moisture predictions, submitted

# Acknowledge

I worked with [Dr. Chaopeng Shen](http://water.engr.psu.edu/shen/index.html) and [MHPI group](https://github.com/mhpi)  until the end of 2019. Please check this [forked repo](https://github.com/mhpi/hydroDL), where MHPI group is carrying on many interesting researches. Here are some papers from MHPI to read and cite:

Feng, DP, K. Fang and CP. Shen, [Enhancing streamflow forecast and extracting insights using continental-scale long-short term memory networks with data integration], Water Resources Reserach (2020), https://doi.org/10.1029/2019WR026793

Shen, CP., [A trans-disciplinary review of deep learning research and its relevance for water resources scientists], Water Resources Research (2018), https://doi.org/10.1029/2018WR022643 

# Example
Two examples with sample data are wrapped up including
 - [train a LSTM network to learn SMAP soil moisture](example/train-lstm.py)
 - [estimate uncertainty of a LSTM network ](example/train-lstm-mca.py)

A demo for temporal test is [here](example/demo-temporal-test.ipynb)

# Document
[description of the SMAP database](document/database-SMAP.py)
[instruction on LSTM uncertainty paper](/documents/SMAP-sigma)
