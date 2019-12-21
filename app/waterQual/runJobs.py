from hydroDL.master import slurm
import os

cmdLine = 'python /home/kuaifang/GitHUB/geolearn/app/waterQual/gridMetMask-job.py -S 0 -E 10'
# slurm.runJob('test', cmdLine)

cmdLine = "screen -dmS test bash -c "+\
    "'srun --exclusive --time 8:0:0 --pty bash;source activate pytorch;" +\
    "python /home/kuaifang/GitHUB/geolearn/app/waterQual/gridMetMask-job.py -S 0 -E 10'"
os.system(cmdLine)
