"""
modified from https://github.com/kkraoj/lfmc_from_sar
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys     
from selenium.webdriver.support.select import Select 
import time
#
url = "https://www.wfas.net/nfmd/public/index.php" # base url for NFMD website

#Getting local session of Chrome
driver=webdriver.Chrome()
 #put here the adress of your page
driver.get(url)
###Download Fuel Moisture Data
driver.find_element(By.CSS_SELECTOR,'#menu_form > input[type="Radio"]:nth-child(4)').click() 

dropdown_css = "#myFormD > table > tbody > tr:nth-child(1) > td:nth-child({0}) > select"
dropdowns = [dropdown_css.format(i) for i in range(1,6)]
# iterate over all levels of like area, state, group, etc..
level = dict(zip(range(1,6), dropdowns))   
pause = 0.1 ## time in seconds to pause between each query. Increase if request gets turned down

d1 = Select(driver.find_element(By.CSS_SELECTOR,level[1]))
for s1 in range(1, len(d1.options)):
    d1.select_by_index(s1)
    time.sleep(pause) 
    driver.find_element(By.CSS_SELECTOR,"#myFormD > table > tbody > tr:nth-child(2) > td > input").click()
