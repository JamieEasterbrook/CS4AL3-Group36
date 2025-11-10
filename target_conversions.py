import json

class TargetConverter:
    def __init__(self):
        self.label_mapping = json.load(open('docs/labels.json', 'r'))['classes']