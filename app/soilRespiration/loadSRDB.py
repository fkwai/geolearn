from hydroDL import kPath, utils

import os
import pandas as pd

dirSRDB = os.path.join(kPath.dirData, 'SRDB', 'SRDB_V5_1827')
fileData = os.path.join(
    dirSRDB, 'data', 'srdb-data-V5.csv', index='Record_number')
df = pd.read_csv(fileData)
