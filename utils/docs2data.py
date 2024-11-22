import os
import pandas as pd
from pathlib import Path
import re
import json

def read_docs_to_dataframe(folder_path):
    # Create an empty list to store the data
    data = []
    
    # Iterate through all files in the specified folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Check if it's a file (not a subdirectory) and has a .txt extension
        if os.path.isfile(file_path) and filename.lower().endswith('.txt'):
            try:
                # Read the content of the file
                with open(file_path, 'r', encoding='utf-8') as file:
                    report = file.read()
                    product_classes = re.findall(r'<em(.*?)>', report, re.DOTALL)
                    product_classes = [f'"{product_class.strip()}"' for product_class in product_classes]
                    pcl = []
                    pcl_len = []
                    product_classes_list = re.findall(r'<em(.*?)>', report, re.DOTALL)
                    for c in product_classes_list:
                        product_classes_list2 = c.split(', ')
                        pcl_len.append(len(product_classes_list2))
                        product_classes_list2 = [f'"{pc.strip()}"' for pc in product_classes_list2]
                        pcl.append(product_classes_list2)
                    report = re.sub(r'<em (.*?)>', '<em>', report)
                    report = re.sub(r'[\n]', ' ', report)
                    excerpts = re.findall(r'<em>(.*?)</em>', report, re.DOTALL)
                    excerpts_list = [f'"{excerpt.strip()}"' for excerpt in excerpts]
                    report = re.sub(r'<em>', '', report)
                    report = re.sub(r'</em>', '', report)
                    count = [len(excerpts), len(product_classes)]
                    count_excerpts = len(excerpts)
                
                # Write answer
                quotes = ""
                for excerpt in excerpts_list:
                    quotes = quotes + str(excerpt) + " "
                quotes = quotes.replace('"', '')

                score = '90'
                if count_excerpts == 0:
                    score = '10'
                else:
                    score = '90'

                answer = [{
                    'quote': quotes,
                    'count_quote': str(count_excerpts),
                    'relevance_score': score,
                    'categories': product_classes
                    }]
                
                # Append the filename and content to the data list
                data.append({
                    'filename': filename,
                    'context': f'"{report.strip()}"',
                    'quote': excerpts_list,
                    'categories': product_classes,
                    #'product_classes_list': pcl,
                    #'products': pcl_len,
                    'count_quote': count_excerpts,
                    'count_categories': count,
                    'answer': answer
                })
            except Exception as e:
                print(f"Error reading file {filename}: {str(e)}")
    
    # Create a pandas DataFrame from the data list
    df = pd.DataFrame(data)
    
    return df