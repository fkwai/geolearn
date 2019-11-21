import urllib


def downloadDaily(siteNo, nwisCode, saveFile):
    strCode = ''.join(['cb_{}=on&'.format(x) for x in nwisCode])
    fmtUrl = 'https://waterdata.usgs.gov/nwis/dv?{}&format=rdb&site_no={}&referred_module=wq&period=&begin_date=2007-10-01&end_date=2019-11-19'
    strUrl = fmtUrl.format(strCode, siteNo)
    urllib.request.urlretrieve(strUrl, saveFile)


def downloadSample(siteNo, saveFile):
    fmtUrl = 'https://nwis.waterdata.usgs.gov/wi/nwis/qwdata/?site_no={}&agency_cd=USGS&inventory_output=0&rdb_inventory_output=file&TZoutput=0&radio_parm_cds=all_parm_cds&qw_attributes=0&format=rdb&qw_sample_wide=separated_wide&rdb_qw_attributes=0&date_format=YYYY-MM-DD'
    strUrl = fmtUrl.format(siteNo)
    urllib.request.urlretrieve(strUrl, saveFile)