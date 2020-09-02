import os
import json
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import pickle as pkl
from nltk.tokenize import word_tokenize, sent_tokenize
from pattern.en import lemma
import random
lemmatizer = WordNetLemmatizer()

verb_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
object_tags = ['NN', 'NNS', 'NNP', 'NNPS']

def load_data(file_path):
    with open(file_path, 'r') as read_file:
        data = json.load(read_file)
    return data

def get_list():
    verb_set = set()
    object_set = set()

    with open('../baseline_classifiers/youcook2_verbs.txt', 'r') as fr:
        for line in fr:
            verb_set.add(lemma(line.rstrip('\r\n')))

    with open('../baseline_classifiers/youcook2_objects.txt', 'r') as fr:
        for line in fr:
            object_set.add(lemmatizer.lemmatize(line.rstrip('\r\n')))

    return verb_set, object_set

def parse_candidates(file_path, verb_set, object_set):
    data = load_data(file_path)
    start_times, end_times, captions = data['start'], data['end'], data['text']
    assert len(start_times) == len(end_times)
    clips = []
    len_clip = 3
    for start_time, end_time, caption in zip(start_times, end_times, captions):
        if end_time - start_time > len_clip:
            clips.append((start_time, end_time, caption))

    random.shuffle(clips)
    return clips

    #     text = word_tokenize(caption)
    #     tags = nltk.pos_tag(text)
    #     lemma_caption = []
    #     for tag in tags:
    #         if tag[1] in verb_tags:
    #             lemma_verb = lemma(tag[0])
    #             if lemma_verb in verb_set:
    #                 lemma_caption.append(lemma_verb)
    #         # if tag[1] in object_tags:
    #         #     lemma_noun = lemmatizer.lemmatize(tag[0])
    #         #     if lemma_noun in object_set:
    #         #         lemma_caption.append(lemma_noun)
    #     if len(lemma_caption) > 0:
    #         count += 1
    #         print(caption, lemma_caption)


if __name__ == '__main__':
    # file_path = './00Lty3r6JLE.json'
    recipes_path = '/home/liangkeg/datasets/howto100m/HowTo100M_recipes.csv'
    verb_set, object_set = get_list()
    caption_root = '/home/liangkeg/datasets/howto100m/captions'
    caption_files = os.listdir(caption_root)

    fw = open('/home/liangkeg/datasets/howto100m/candidate_clips.txt', 'w')
    with open(recipes_path, 'r') as fr:
        for line in fr:
            file_name = line.split(',')[0]
            caption_path = os.path.join(caption_root, '{}.json'.format(file_name))
            clips = parse_candidates(caption_path, verb_set, object_set)
            # start_time, end_time, caption = parse_candidates(caption_path, verb_set, object_set)
            if len(clips) > 0:
                start_time, end_time, caption = clips[0]
                print(file_name, start_time, end_time, caption)
                fw.write('{},{},{},{}\n'.format(file_name, start_time, end_time, caption))