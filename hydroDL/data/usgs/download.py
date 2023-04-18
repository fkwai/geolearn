import urllib.request
__all__ = ['downloadDaily', 'downloadSample','downloadHourly']


def downloadDaily(siteNo, nwisCode, saveFile, state=None):
    # update - USGS (seems) does not require state now
    strCode = ''.join(['cb_{}=on&'.format(x) for x in nwisCode])
    if state is None:
        fmtUrl = 'https://waterdata.usgs.gov/nwis/dv?{}&format=rdb&site_no={}&period=&begin_date=1900-01-01'
        strUrl = fmtUrl.format(strCode, siteNo)
    else:
        fmtUrl = 'https://waterdata.usgs.gov/{}/nwis/dv?{}&format=rdb&site_no={}&period=&begin_date=1900-01-01'
        strUrl = fmtUrl.format(state, strCode, siteNo)
    urllib.request.urlretrieve(strUrl, saveFile)
    return strUrl

def downloadHourly(siteNo, saveFile):
    # update - USGS (seems) does not require state now
    # move to waterservices, download all code (00060, 00065)    
    fmtUrl ='https://waterservices.usgs.gov/nwis/iv/?sites={}&startDT=1900-01-01&siteStatus=all&format=rdb'    
    strUrl = fmtUrl.format(siteNo)    
    urllib.request.urlretrieve(strUrl, saveFile)
    return strUrl


def downloadSample(siteNo, saveFile, state=None):
    if state is None:
        fmtUrl = 'https://nwis.waterdata.usgs.gov/nwis/qwdata/?site_no={}&agency_cd=USGS&inventory_output=0&rdb_inventory_output=file&TZoutput=0&radio_parm_cds=all_parm_cds&qw_attributes=0&format=rdb&qw_sample_wide=separated_wide&rdb_qw_attributes=0&date_format=YYYY-MM-DD'
        strUrl = fmtUrl.format(siteNo)
    else:
        fmtUrl = 'https://nwis.waterdata.usgs.gov/{}/nwis/qwdata/?site_no={}&agency_cd=USGS&inventory_output=0&rdb_inventory_output=file&TZoutput=0&radio_parm_cds=all_parm_cds&qw_attributes=0&format=rdb&qw_sample_wide=separated_wide&rdb_qw_attributes=0&date_format=YYYY-MM-DD'
        strUrl = fmtUrl.format(state, siteNo)
    urllib.request.urlretrieve(strUrl, saveFile)
    return strUrl
