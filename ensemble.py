import json
import os
from collections import defaultdict,OrderedDict

"""nbest_predictions n개를 앙상블하는 코드입니다. """

path = '.'
paths = os.listdir(path)
full_path_list = []
answers = {}
for i in range(1,5):
    with open(f'nbest_predictions{i}.json') as f:
        json_data = json.load(f)
    for idx,j in json_data.items():
        if idx not in answers:
            answers[idx] = {}
        for item in j:
            try:
                answers[idx][item['text']] += item['probability']
            except:
                answers[idx][item['text']] = item['probability']

submission = {}
for key, value in answers.items():
    list_sort = sorted(list(value), key=lambda x: answers[key][x])
    submission[key] = list_sort[-1]
with open('submisson.json','w') as f:
    json.dump(submission,f)