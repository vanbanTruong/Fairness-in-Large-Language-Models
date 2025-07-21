import csv
import json

def load_crows_pairs_dataset():
    '''
    Extract stereotypical and anti-stereotypical sentences from crows-paris.
    '''
    data = []

    with open('data\crow_pairs_dataset.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            example = {}
            direction = row['stereo_antistereo']
            example['direction'] = direction
            example['bias_type'] = row['bias_type']

            example['stereotype'] = row['sent_more']
            example['anti-stereotype'] = row['sent_less']
            data.append(example)

    return data


def load_stereoset_dataset():
    '''
    Extract stereotypical and anti-stereotypical sentences from StereoSet.
    '''
    data = []
    data = []

    with open('data/stereoset_dataset.json') as f:
        input = json.load(f)
        for annotations in input['data']['intrasentence']:
            example = {}
            example['bias_type'] = annotations['bias_type']
            for annotation in annotations['sentences']:
                gold_label = annotation['gold_label']
                sentence = annotation['sentence']
                example[gold_label] = sentence
            data.append(example)

    return data