import os
import requests
import json
import pandas as pd

def fetch_and_append_air_quality_data(excel_path):
    url = ''
    
    response = requests.get(url)
    
    if response.status_code == 200:
        data_dict = json.loads(response.text)
        
        if data_dict['status'] == 'ok':
            entries = data_dict.get('entries', [])
            exclusions = data_dict.get('exclusion', [])
            
            entries = entries if isinstance(entries, list) else []
            exclusions = exclusions if isinstance(exclusions, list) else []

            data_for_excel = []
            
            for entry in entries + exclusions:
                place_dict = {key: entry.get(key, None) for key in ['pm25', 'pm10', 'pm1', 'co2', 'hcho', 'tvoc', 'co', 't', 'h', 'status', 'time', 'name']}
                data_for_excel.append(place_dict)

            new_data_df = pd.DataFrame(data_for_excel)

            if os.path.exists(excel_path):
                old_data_df = pd.read_excel(excel_path)
                all_data_df = pd.concat([old_data_df, new_data_df])
                final_data_df = all_data_df.drop_duplicates(subset=['name', 'time'], keep='last')
                final_data_df.to_excel(excel_path, index=False)
            else:
                new_data_df.to_excel(excel_path, index=False)
            
            return f'Data appended to {excel_path}.'
        else:
            return 'Data status not ok.'
    else:
        return 'API request failed.'

fetch_and_append_air_quality_data('air_quality_data.xlsx')
