import os
import json
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import pickle as pkl
from nltk.tokenize import word_tokenize, sent_tokenize
import matplotlib as mpl
mpl.rcParams.update({'font.size': 4})

mpl.use('Agg')
import matplotlib.pyplot as plt

lemmatizer = WordNetLemmatizer()

verb_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
object_tags = ['NN', 'NNS', 'NNP', 'NNPS']

def load_data(file_path):
    with open(file_path, 'r') as read_file:
        data = json.load(read_file)
    return data

def create_labels(caption, verb_set, object_set):

    text = word_tokenize(caption)
    tags = nltk.pos_tag(text)
    verbs = []
    objs = []
    for tag in tags:
        lemma = lemmatizer.lemmatize(tag[0])
        if tag[1] in verb_tags and lemma in verb_set:
            verbs.append(lemma)
        if tag[1] in object_tags and lemma in object_set:
            objs.append(lemma)
    return verbs, objs

def get_list():
    verb_set = set()
    object_set = set()

    with open('./youcook2_verbs.txt', 'r') as fr:
        for line in fr:
            verb_set.add(lemmatizer.lemmatize(line.rstrip('\r\n')))

    with open('./youcook2_objects.txt', 'r') as fr:
        for line in fr:
            object_set.add(lemmatizer.lemmatize(line.rstrip('\r\n')))

    return verb_set, object_set

def convert_to_labels(token_list, token_dict):
    labels = len(token_dict.keys()) * [0]
    for token in token_list:
        labels[token_dict[token]] = 1
    return labels


if __name__ == '__main__':
    feat_type = '3d'
    # data_split = 'training'
    data_split = 'validation'

    verb_set, object_set = get_list()
    verb_list = sorted(list(verb_set))
    object_list = sorted(list(object_set))

    verb_dict = {key:value for value, key in enumerate(verb_list)}
    object_dict = {key:value for value, key in enumerate(object_list)}

    print(verb_dict)
    # print(object_dict)
    ann_path = '/home/liangkeg/third_hand/data/youcookii_annotations_trainval.json'
    dst_root = '/home/liangkeg/third_hand/data/YouCookII/features/multilabel_features'
    feat_root = '/home/liangkeg/third_hand/data/YouCookII/features/{}'.format(feat_type)
    anns = load_data(ann_path)

    video_infos = anns['database']
    if feat_type == '2d':
        frame_rate = 1.0
    else:
        frame_rate = 24.0/16.0

    verb_features = []
    verb_labels = []
    object_features = []
    object_labels = []

    stat_verbs = {}
    verb_count = 0

    stat_nouns = {}
    noun_count = 0

    for video_name in video_infos:
        video_info = video_infos[video_name]
        subset = video_info['subset']
        if subset != data_split:
            continue
        feat_path = os.path.join(feat_root, '{}.npy'.format(video_name))
        if not os.path.exists(feat_path):
            continue
        features = np.load(feat_path)
        for annotations in video_info['annotations']:
            start_time, end_time = annotations['segment']
            caption = annotations['sentence']
            verbs, objs = create_labels(caption, verb_set, object_set)
            start_index = int(start_time * frame_rate)
            end_index = int(end_time * frame_rate) + 1
            # feature = np.mean(features[start_index:end_index], axis=0, keepdims=True)
            feature = np.mean(features[start_index:end_index], axis=0)
            if len(verbs) != 0:
                # print(caption, verbs)
                if len(verbs) > 1:
                    verb_count += 1
                for verb in verbs:
                    stat_verbs[verb] = stat_verbs.get(verb, 0) + 1
                verb_label = convert_to_labels(verbs[:1], verb_dict)
                verb_features.append(feature)
                # verb_labels.append(verb_label)
                verb_labels.append(verb_dict[verbs[0]])

            if len(objs) != 0:
                if len(objs) > 1:
                    noun_count += 1
                for obj in objs:
                    stat_nouns[obj] = stat_nouns.get(obj, 0) + 1
                object_label = convert_to_labels(objs, object_dict)
                object_features.append(feature)
                object_labels.append(object_label)

    verb_features = np.asarray(verb_features)
    verb_labels = np.asarray(verb_labels)
    object_features = np.asarray(object_features)
    object_labels = np.asarray(object_labels)


    verb_data = {}
    verb_data['features'] = verb_features
    verb_data['labels'] = verb_labels

    object_data = {}
    object_data['features'] = object_features
    object_data['labels'] = object_labels

    # verb_path = os.path.join(dst_root, 'verb_{}_{}.pkl'.format(feat_type, data_split))
    # with open(verb_path,'wb') as output_file:
    #     pkl.dump(verb_data, output_file)

    # object_path = os.path.join(dst_root, 'object_{}_{}.pkl'.format(feat_type, data_split))
    # with open(object_path, 'wb') as output_file:
    #     pkl.dump(object_data, output_file)