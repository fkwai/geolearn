from hydroDL.master import slurm
import os

""" extract data from gridMET """
varLst = ['pr', 'sph', 'srad', 'tmmn', 'tmmx', 'pet', 'etr']
yrLst = list(range(1979, 1980))

for yr in yrLst:
    for var in varLst:
        # cmd = 'python /home/kuaifang/GitHUB/geolearn/app/waterQual/gridMetExtract.py -var {} -yr {}'
        cmd = 'python /home/users/kuaifang/GitHUB/geolearn/app/waterQual/gridMetExtract.py -var {} -yr {}'
        cmdLine = cmd.format(var, yr)
        jobName = '{}_{}'.format(yr, var)
        slurm.submitJob(jobName, cmdLine, nH=1, nM=4)

# cmdLine = "screen -dmS test bash -c "+\
#     "'srun --exclusive --time 8:0:0 --pty bash;source activate pytorch;" +\
#     "python /home/kuaifang/GitHUB/geolearn/app/waterQual/gridMetMask-job.py -S 0 -E 10'"
# os.system(cmdLine)
