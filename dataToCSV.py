import os
import csv
import json

clicks = []

for clickType in os.listdir('clicks/'):
    for file in os.listdir('clicks/' + clickType):
        with open('clicks/' + clickType + '/' + file, 'r') as file:
            data = ''
            for click in json.loads(file.read()):
                data += str(click) + '|'

            clicks.append(f'{data},{clickType}')
            print(f'Appended {file.name}')

with open('data.csv', 'w', newline = '') as csvfile:
    writer = csv.writer(csvfile, delimiter = '\n')
    writer.writerow(clicks)