import json

label_json:dict = json.load(open('docs/labels.json'))
raw:list[str] = label_json['original']
separated:set[str] = set()

for labels in raw:
    label_list = labels.split(',')
    for label in label_list:
        separated.add(label)

label_json['separated'] = list(separated)
json.dump(label_json, open('docs/labels.json', 'w'))