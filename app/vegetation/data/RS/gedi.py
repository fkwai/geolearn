import requests as r
from datetime import datetime
import os


def gedi_finder(product, bbox):

    # Define the base CMR granule search url, including LPDAAC provider name and max page size (2000 is the max allowed)
    cmr = "https://cmr.earthdata.nasa.gov/search/granules.json?pretty=true&provider=LPDAAC_ECS&page_size=2000&concept_id="

    # Set up dictionary where key is GEDI shortname + version
    concept_ids = {
        'GEDI01_B.002': 'C1908344278-LPDAAC_ECS',
        'GEDI02_A.002': 'C1908348134-LPDAAC_ECS',
        'GEDI02_B.002': 'C1908350066-LPDAAC_ECS',
    }

    # CMR uses pagination for queries with more features returned than the page size
    page = 1
    bbox = bbox.replace(' ', '')  # remove any white spaces
    try:
        # Send GET request to CMR granule search endpoint w/ product concept ID, bbox & page number, format return as json
        cmr_response = r.get(
            f"{cmr}{concept_ids[product]}&bounding_box={bbox}&pageNum={page}"
        ).json()['feed']['entry']

        # If 2000 features are returned, move to the next page and submit another request, and append to the response
        while len(cmr_response) % 2000 == 0:
            page += 1
            cmr_response += r.get(
                f"{cmr}{concept_ids[product]}&bounding_box={bbox}&pageNum={page}"
            ).json()['feed']['entry']

        # CMR returns more info than just the Data Pool links, below use list comprehension to return a list of DP links
        return [c['links'][0]['href'] for c in cmr_response]
    except:
        # If the request did not complete successfully, print out the response from CMR
        print(
            r.get(
                f"{cmr}{concept_ids[product]}&bounding_box={bbox.replace(' ', '')}&pageNum={page}"
            ).json()
        )


product = 'GEDI02_B.002'
bbox = '-73.65,-12.64,-47.81,9.7'
granules = gedi_finder(product, bbox)

# find corresponding lat lon for 500m box

lat = 46.25972222222222
lon = -124.1327777777778
r1 = 6378137
r2 = 6356752.3142
import math
rlon = math.cos(lat * math.pi / 180) * r1
dlon = 500 / (2 * math.pi * rlon) * 360
rlat = r2 / math.sqrt(1 - (1 - r2 / r1) * math.sin(lat * math.pi / 180) ** 2)
dlat = 500 / (2 * math.pi * rlat) * 360

bbox= f'{lon-dlon},{lat-dlat},{lon+dlon},{lat+dlat}'
bbox=[lon-dlon,lat-dlat,lon+dlon,lat+dlat]
product = 'GEDI02_B.002'
granules = gedi_finder(product, bbox)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

file='/home/kuai/GitHUB/pyGEDI-master/notebook/data/formats/Box_GEDI02_B.csv'
df=pd.read_csv(file)

df['latitude']=df['latitude']+60
df['longitude']=df['longitude']-80

fig,ax=plt.subplots(1,1)
sc=ax.scatter(df['longitude'],df['latitude'],s=5,c=df['Height'])
# plot bounding box
bbox=[lon-dlon,lat-dlat,lon+dlon,lat+dlat]

ax.plot([bbox[0],bbox[2],bbox[2],bbox[0],bbox[0]],[bbox[1],bbox[1],bbox[3],bbox[3],bbox[1]])
plt.colorbar(sc)
fig.show()