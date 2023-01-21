from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.select import Select
import time
#
url = "https://www.wfas.net/nfmd/public/index.php"  # base url for NFMD website

# Getting local session of Chrome
driver = webdriver.Chrome()
# driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
# put here the adress of your page
driver.get(url)
# Download Fuel Moisture Data
driver.find_element(By.CSS_SELECTOR,
    '#menu_form > input[type="Radio"]:nth-child(4)').click()

dropdown_css = "#myFormD > table > tbody > tr:nth-child(1) > td:nth-child({0}) > select"
dropdowns = [dropdown_css.format(i) for i in range(1, 6)]
# iterate over all levels of like area, state, group, etc..
level = dict(zip(range(1, 6), dropdowns))
pause = 0.1  # time in seconds to pause between each query. Increase if request gets turned down
d1 = Select(driver.find_element(By.CSS_SELECTOR,level[1]))
for s1 in range(1, len(d1.options)):
    d1.select_by_index(s1)
    time.sleep(pause)
    d2 = Select(driver.find_element(By.CSS_SELECTOR,level[2]))
    for s2 in range(1, len(d2.options)):
        d2.select_by_index(s2)
        time.sleep(pause)
        d3 = Select(driver.find_element(By.CSS_SELECTOR,level[3]))
        for s3 in range(1, len(d3.options)):
            d3.select_by_index(s3)
            time.sleep(pause)
            d4 = Select(driver.find_element(By.CSS_SELECTOR,level[4]))
            for s4 in range(1, len(d4.options)):
                d4.select_by_index(s4)
                time.sleep(pause)
                d5 = Select(driver.find_element(By.CSS_SELECTOR,level[5]))
                for s5 in range(1, len(d5.options)):
                    d5.select_by_index(s5)
                    time.sleep(pause)
                    driver.find_element(By.CSS_SELECTOR,
                        "#myFormD > table > tbody > tr:nth-child(2) > td > input").click()
