import csv
import json

def make_json(csvFilePath,jsonFilePath):
    data = {}

    with open(csvFilePath, encoding='utf-8') as csvf:
        csvReader = csv.DictReader(csvf)
        i = 0
        for rows in csvReader:
            key = i
            data[key] = rows
            i+=1
            
 
    with open(jsonFilePath, 'w', encoding='utf-8') as jsonf:
        jsonf.write(json.dumps(data, indent=4))

for i in range(0,14):
    jsonFilePath = '/Users/sheshmani/Desktop/community_rating/data_dir/split_json/'+str(i)+'.json'
    csvFilePath = '/Users/sheshmani/Desktop/community_rating/homes_data/split_csv_files/'+str(i)+'.csv'
    make_json(csvFilePath,jsonFilePath)
