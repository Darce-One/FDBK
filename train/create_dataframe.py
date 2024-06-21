import pandas as pd
import os
import numpy as np
import json


folder_path = 'dataset_final'
json_file_path = 'params_final.json'
csv_file_path = 'fdbk_dataframe_final.csv'


file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]


with open(json_file_path, 'r') as json_file:
    json_data = json.load(json_file)
    parameters = json_data.get('data', {})
    print(parameters)


data = []
for file_path in file_paths:
    filename = os.path.basename(file_path)
    relative_name = f"example-{filename.split('.')[0].split('-')[-1]}"
    #print(relative_name)
    #print(filename)
    param = parameters.get(relative_name, None)
    print(param)
    if param is not None:
        entry = {'file_path': file_path}
        for i, p in enumerate(param):
            entry[f'parameter_{i+1}'] = p
        data.append(entry)

df = pd.DataFrame(data)
print(df.head())


df.to_csv(csv_file_path, index=False)
