import nltk
# import these modules
# nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
lemmatizer = WordNetLemmatizer()

verb_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
object_tags = ['NN', 'NNS', 'NNP', 'NNPS']

def create_labels(captions, verb_set, object_set):
    caption_verbs = []
    caption_objs = []

    for caption in captions:
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
        caption_verbs.append(verbs)
        caption_objs.append(objs)
    return caption_verbs, caption_objs


def get_list():
    verb_set = set()
    object_set = set()

    with open('./youcook2_verbs.txt', 'r') as fr:
        for line in fr:
            verb_set.add(lemmatizer.lemmatize(line.rstrip('\r\n')))

    with open('./youcook2_objects.txt', 'r') as fr:
        for line in fr:
            object_set.add(lemmatizer.lemmatize(line.rstrip('\r\n')))

    print(len(verb_set), len(object_set))
    return verb_set, object_set

if __name__ == '__main__':

    video_ids = []
    start_frames = []
    end_frames = []
    captions = []
    split_types = []
    with open('./youcook_clip_info.txt', 'r') as fr:
        for line in fr:
            video_id, start_frame, end_frame, caption, _, _, split_type = line.rstrip('\r\n').split(',')

            video_ids.append(video_id)
            start_frames.append(start_frame)
            end_frames.append(end_frame)
            captions.append(caption)
            split_types.append(split_type)

    verb_set, object_set = get_list()
    caption_verbs, caption_objs = create_labels(captions, verb_set, object_set)
    print(caption_verbs, caption_objs)



    # print(lemmatizer.lemmatize('took'))