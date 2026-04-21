import csv
import numpy as np

output = []

with open('Testcsv(in).csv', mode='r', newline='', encoding='utf-8-sig') as file:
    reader = csv.reader(file)
    
    for row in reader:
        if row:
            # clean string (removes stray commas/spaces)
            cleaned = row[0].replace(',', '')
            
            # fast conversion from semicolon-separated values
            arr = np.fromstring(cleaned, sep=';')
            
            output.append(arr)

output = np.array(output, dtype=object)

print(output)