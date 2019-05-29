from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
import time
import threading

#Make a certain number of particular submissions autonomously and as quickly as the network allows and record the corresponding rmse

def hak(n):
    options = webdriver.ChromeOptions()
    options.add_argument('headless')

    driver = webdriver.Chrome(
        "C:/Program Files/Wolfram Research/Wolfram CDF Player/11.3/SystemFiles/Components/WebUnit/Resources/DriverBinaries/ChromeDriver/Windows-x86-64/chromedriver.exe",options=options)
    for i in range(0,n):
        driver.get('http://cs156.caltech.edu/scoreboard/')

        #inputs = driver.find_elements_by_tag_name("input")

        #print(len(inputs))

        username = driver.find_element_by_id("id_teamid")
        file = driver.find_element_by_id("id_file")

        username.send_keys("piavwmto")
        k=j
        j = j + 1
        file.send_keys("C:/Users/DFCTech/Downloads/mu/spf"+k+".dta")

        driver.find_elements_by_tag_name("input")[4].click()

        delay = 10 # seconds
        try:
            myElem = WebDriverWait(driver, delay).until(EC.presence_of_element_located((By.TAG_NAME, 'h3')))
            print(time.time()-start)
            rs = driver.find_elements_by_tag_name('h3')[1].text
            file.write(k + "\t" + rs)
            print(rs)


        except TimeoutException:
            print("Loading took too much time!")

N_THREADS = 20
N_RECOVER_PER_THREAD = 10;
ts = []
j=0
file = open("C:\out.txt","w")
for i in range(0, N_THREADS):
    ts.append(threading.Thread(target=hak,args=(N_RECOVER_PER_THREAD,)))

for tn in ts:
    tn.start()

start = time.time()
print("start")

for tn in ts:
    tn.join()

end = time.time()
print(end - start)
