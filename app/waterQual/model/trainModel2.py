from hydroDL.master import basins
basins.trainModelTS('basinAll', 'first80')

# # predict - point-by-point
# yOut = trainTS.testModel(model, x, xc)
# q, c = wqData.transOut(yOut[:, :, :ny], yOut[-1, :, ny:], statY, statYC)
