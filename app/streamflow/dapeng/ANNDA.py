from hydroDL import pathCamels, master, utils
from hydroDL.master import default
from hydroDL.post import plot, stat
from hydroDL.model import crit
import matplotlib.pyplot as plt
from hydroDL.data import camels
import torch
import numpy as np
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from time import time
# Hyperparameters
nDay=1
EPOCH=200
BATCH_SIZE=365*100
Hid_lay=[256,256]
expname = 'hid1_'+str(Hid_lay[0])+'hid2_'+str(Hid_lay[1])+'batch_'+str(BATCH_SIZE)+'epoch_'+str(EPOCH)
savepath = pathCamels['Out'] + '/comparison/ANN/' + expname+'/addstatics1'
if not os.path.isdir(savepath):
    os.makedirs(savepath)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Import data
optData = default.update(default.optDataCamels, daObs=nDay, rmNan=[True, True])
df, x, y, c = master.master.loadData(optData)
c = np.expand_dims(c, axis=1)
c = np.tile(c, [1, x.shape[1], 1])
x = np.concatenate((x, c), axis=2)
x2 = torch.from_numpy(np.reshape(x,(-1, x.shape[2])).astype('float32'))
y2 = torch.from_numpy(np.reshape(y,(-1, y.shape[2])).astype('float32'))
train_set =  TensorDataset(x2, y2)
opttestData = default.update(default.optDataCamels, daObs=nDay, tRange=[19950101, 20000101])
dftest, xt, yt, c = master.master.loadData(opttestData)
c = np.expand_dims(c, axis=1)
c = np.tile(c, [1, xt.shape[1], 1])
xt = np.concatenate((xt, c), axis=2)
xt2 = torch.from_numpy(np.reshape(xt,(-1, xt.shape[2])).astype('float32'))
yt2 = torch.from_numpy(np.reshape(yt,(-1, yt.shape[2])).astype('float32'))
test_set =  TensorDataset(xt2, yt2)
ngage = x.shape[0]
daylen = xt.shape[1]
Pred = np.full(yt.shape, np.nan)
train_loader=DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader=DataLoader(dataset=test_set, batch_size=daylen*100, shuffle=False)

# define model
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden[0])   # hidden layer1
        self.hidden2 = torch.nn.Linear(n_hidden[0], n_hidden[1])
        self.predict = torch.nn.Linear(n_hidden[1], n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.predict(x)
        return x

net = Net(n_feature=x.shape[2], n_hidden=Hid_lay, n_output=1)     # define the network
net.to(device,dtype=torch.float32)
print(net)

# define the training model
optimizer = torch.optim.Adadelta(net.parameters())
loss_func = crit.RmseLoss(get_length=True) # this is for regression mean squared loss
logger = {}
logger['rmse_train'] = []
logger['rmse_test'] = []
logger['train_loss']=[]

# define testing
def test(epoch):
    testmse = 0.0
    Nsample = 0
    for step, (batch_x, batch_y) in enumerate(test_loader):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        out = net(batch_x)
        loss, Nday = loss_func(out, batch_y)
        tempmse = (loss.item() ** 2) * Nday
        testmse += tempmse
        Nsample += Nday
    rmse = np.sqrt(testmse / Nsample)  # unit:cm
    print("epoch: {}, test RMSE: {:.6f}".format(epoch, rmse))
    logger['rmse_test'].append(rmse)

# traning
print('Start training.......................................................')
tic = time()
for epoch in range(EPOCH):
    mse = 0.0
    Nsample = 0
    for step, (batch_x, batch_y) in enumerate(train_loader):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        out=net(batch_x)
        loss, Nday=loss_func(out, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tempmse = (loss.item()**2)*Nday
        mse += tempmse
        Nsample += Nday
        logger['train_loss'].append(loss.item())
        if (step+1) % 100 == 0:
            print('epoch{}-step:{}, loss: {:.6f}'.format(epoch, step + 1, loss))
    rmse = np.sqrt(mse/Nsample)
    print("epoch: {}, training RMSE: {:.6f}".format(epoch, rmse))
    logger['rmse_train'].append(rmse)
    # test
    with torch.no_grad():
        test(epoch)
tic2 = time()
print("Finish training/testing using {} seconds".format(tic2 - tic))

# Save variables and get plots
save_name=savepath+'/ANN.pkl'
torch.save(net,save_name)
x_axis = np.arange(1, EPOCH+1)
rmse_train = logger['rmse_train']
rmse_test = logger['rmse_test']
plt.figure()
plt.plot(x_axis, rmse_train, label="Train: {:.3f}".format(np.mean(rmse_train[-5:])))
plt.plot(x_axis, rmse_test, label="Test: {:.3f}".format(np.mean(rmse_test[-5:])))
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.legend(loc='upper right')
plt.savefig(savepath + "/rmse.pdf", dpi=600)
np.savetxt(savepath + "/rmse_train.txt", rmse_train)
np.savetxt(savepath + "/rmse_test.txt", rmse_test)
# plot loss process of training
train_loss=logger['train_loss']
x_axis=np.arange(1,len(train_loss)+1)
plt.figure()
plt.plot(x_axis, train_loss, label="Train loss: {:.3f}".format(np.mean(train_loss[-5:])))
plt.xlabel('Steps')
plt.ylabel('Batch Loss')
plt.legend(loc='upper right')
plt.savefig(savepath + "/trainloss.pdf", dpi=600)


# make predictions for each basin
# load model
net = torch.load(savepath+'/ANN.pkl')
net.to(device,dtype=torch.float32)
net.eval()
for ii in range(ngage):
    xdata = torch.from_numpy(xt[ii, :,:].astype('float32'))
    xdata = xdata.to(device)
    ypred = net(xdata).detach().cpu()
    Pred[ii, :, 0] = ypred.numpy().squeeze()
pred = camels.transNorm(Pred, 'usgsFlow', toNorm=False)
obs = camels.transNorm(yt, 'usgsFlow', toNorm=False)
gageid = 'All'
pred = camels.basinNorm(pred, gageid=gageid, toNorm=False)
obs = camels.basinNorm(obs, gageid=gageid, toNorm=False)
# plot box
statDictLst = [stat.statError(pred.squeeze(), obs.squeeze())]
keyLst=['Bias', 'RMSE', 'NSE']
dataBox = list()
for iS in range(len(keyLst)):
    statStr = keyLst[iS]
    temp = list()
    for k in range(len(statDictLst)):
        data = statDictLst[k][statStr]
        data = data[~np.isnan(data)]
        temp.append(data)
    dataBox.append(temp)
# plt.style.use('classic')
plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams["legend.columnspacing"]=0.1
plt.rcParams["legend.handletextpad"]=0.2
labelname = ['ANN']
nDayLst = [1]
for nDay in nDayLst:
    labelname.append('DA-'+str(nDay))
xlabel = ['Bias ($\mathregular{ft^3}$/s)', 'RMSE ($\mathregular{ft^3}$/s)', 'NSE']
fig = plot.plotBoxFig(dataBox, xlabel, labelname, sharey=False, figsize=(10, 5))
fig.patch.set_facecolor('white')
fig.show()
# write evaluation results
mFile = os.path.join(savepath, 'evaluation.npy')
if not os.path.isdir(savepath):
    os.makedirs(savepath)
np.save(mFile, statDictLst[0])
