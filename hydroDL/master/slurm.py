import os


def runJob(jobName, cmdLine):
    jobFile = os.path.join(r'/home/kuaifang/jobs', jobName)
    with open(jobFile,'w') as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines("#SBATCH --job-name=%s.job\n" % jobName)
        fh.writelines("#SBATCH --output=.out/%s.out\n" % jobName)
        fh.writelines("#SBATCH --error=.out/%s.err\n" % jobName)
        fh.writelines("#SBATCH --time=2:0:0\n")
        fh.writelines("#SBATCH --mem=20000\n")
        fh.writelines("#SBATCH --qos=normal\n")
        fh.writelines("#SBATCH --mail-type=ALL\n")
        fh.writelines("#SBATCH --mail-user=$USER@stanford.edu\n")
        fh.writelines("source activate pytorch\n")
        fh.writelines(cmdLine)
    os.system("sbatch %s" %jobFile)

