from hydroDL.master import basins
basins.trainModelTS('temp10', 'first80', saveEpoch=5, nEpoch=20)

# # predict - point-by-point
# yOut = trainTS.testModel(model, x, xc)
# q, c = wqData.transOut(yOut[:, :, :ny], yOut[-1, :, ny:], statY, statYC)
