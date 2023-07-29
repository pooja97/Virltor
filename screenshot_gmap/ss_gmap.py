import json
import os
from glob import glob
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time
from tqdm import tqdm
import pandas as pd
import requests

import io

source_dir = '/Users/sheshmani/Desktop/virltor/data_dir'
jsonlist = glob(os.path.join(source_dir,"split_json",'*.json'))
txt_download_path = '/Users/sheshmani/Desktop/virltor/data_dir/download_record.txt'
txt_json_path = '/Users/sheshmani/Desktop/virltor/data_dir/update_json_record.txt'

# Start the chrome browser
s=Service(ChromeDriverManager().install())
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--incognito")
driver = webdriver.Chrome(service=s, options=chrome_options)
driver.maximize_window()
time.sleep(5)

# screenshot at specific location
def screenshot_gmap(loc_id, gmap_path, view):
    screenshot_savepath = os.path.join(gmap_path, loc_id + '_' + view + '.png')
    driver.save_screenshot(screenshot_savepath)


# This method will open https://maps.google.com, search for the street view and screenshot
def search_gmap(doc_detail_j, gmap_path):
    loc = str(doc_detail_j['latitude']) + ' ' + str(doc_detail_j['longitude'])
    loc_id = doc_detail_j['file_name']
    driver.get('https://maps.google.com')
    time.sleep(5)
    try:
        driver.find_element(By.XPATH, "//input[contains(@id, 'searchboxinput')]").click()
        driver.find_element(By.XPATH, "//input[contains(@id, 'searchboxinput')]").send_keys(loc)
        time.sleep(2)
        driver.find_element(By.XPATH, "//span[text()= '" + loc + "' ]").click()
        time.sleep(5)
        driver.find_element(By.XPATH, "//div[contains(@class, 'd6JfQc')]").click()
        time.sleep(8)

        # screenshot
        driver.find_element(By.XPATH, "//button[contains(@aria-label, 'Reset the view')]").click()
        time.sleep(3)
        screenshot_gmap(loc_id, gmap_path, '0')
        time.sleep(1)

        driver.find_element(By.XPATH, "//button[contains(@aria-label, 'Rotate the view clockwise')]").click()
        time.sleep(2)
        screenshot_gmap(loc_id, gmap_path, '1')
        time.sleep(1)

        driver.find_element(By.XPATH, "//button[contains(@aria-label, 'Rotate the view clockwise')]").click()
        time.sleep(2)
        screenshot_gmap(loc_id, gmap_path, '2')
        time.sleep(1)

        driver.find_element(By.XPATH, "//button[contains(@aria-label, 'Rotate the view clockwise')]").click()
        time.sleep(2)
        screenshot_gmap(loc_id, gmap_path, '3')
        time.sleep(1)

        # Image capture date
        doc_detail_j['street_view'] = 'Yes'
        try:
            doc_detail_j['image_capture_date'] = driver.find_element(By.XPATH, "//span[text()[contains(.,'Image capture')]]").text
        except Exception as e:
            doc_detail_j['image_capture_date'] = 'No'
            with open(txt_download_path, "a") as txt_file:
                txt_file.write('can download four view images for: ' + loc_id + ', but fail to get image capture date' + "\n")
        return doc_detail_j

    except Exception as e:
        with open(txt_download_path, "a") as txt_file:
            txt_file.write('Unable to get four views for: ' + loc_id + "\n")
        doc_detail_j['street_view'] = 'No'
        doc_detail_j['image_capture_date'] = 'No'
        return doc_detail_j

for i in range(1,len(jsonlist)+1):
    # check path exist
    gmap_path = os.path.join(source_dir, str(i))
    if os.path.exists(gmap_path):
        pass
    else:
        os.makedirs(gmap_path)

    # read json file
    json_file = os.path.join(source_dir,"split_json",str(i) + '.json')
    with open(json_file, 'r') as jf:
        data = json.load(jf)
    jf.close()
    doc_detail = data["document_details"]
    # screenshot and update json
    doc_detail_screenshot = []
    for j in tqdm(range(len(doc_detail))):
        doc_detail_j = doc_detail[str(j)]
        # print(doc_detail_j)
        # loc = [doc_detail_j['LATITUDE'], doc_detail_j['LONGITUDE']]
        # loc = str(doc_detail_j['LATITUDE']) + ' ' + str(doc_detail_j['LONGITUDE'])
        # loc_id = doc_detail_j['file_name']
        doc_detail_j_update = search_gmap(doc_detail_j, gmap_path)
        doc_detail_screenshot.append(doc_detail_j_update)



    # count updated json
    with open(txt_json_path, "a") as txt_file:
        txt_file.write(str(i) + '_update.json length: ' + str(len(doc_detail_screenshot)) + "\n")

    # write updated json
    json_update_file = os.path.join(source_dir, "update_json", str(i) + '_update.json')
    with open(json_update_file, 'w') as juf:
        json.dump({'document_details': doc_detail_screenshot
                   }, juf)
    juf.close()

