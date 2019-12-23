import os


def submitJob(jobName, cmdLine, nH=8, nM=16):
    jobFile = os.path.join(r'/home/kuaifang/jobs', jobName)
    with open(jobFile, 'w') as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines('#SBATCH --job-name={}\n'.format(jobName))
        fh.writelines('#SBATCH --output={}.out\n'.format(jobFile))
        fh.writelines('#SBATCH --error={}.err\n'.format(jobFile))
        fh.writelines('#SBATCH --time={}:0:0\n'.format(nH))
        fh.writelines('#SBATCH --mem={}000\n'.format(nM))
        fh.writelines('#SBATCH --qos=normal\n')
        fh.writelines('#SBATCH --mail-type=ALL\n')
        fh.writelines('#SBATCH --mail-user=$USER@stanford.edu\n')
        fh.writelines('source activate pytorch\n')
        fh.writelines(cmdLine)
    os.system("sbatch %s" % jobFile)
