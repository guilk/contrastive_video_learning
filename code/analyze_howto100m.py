import json
import pandas
import os

if __name__ == '__main__':
    file_path = '../../../data/metadata/caption.json'
    caption_folder = '../../../data/captions'
    if not os.path.exists(caption_folder):
        os.makedirs(caption_folder)
    # df = pandas.read_json(file_path, lines=True)
    # print(type(df))

    with open(file_path) as input_file:
        # print(type(input_file))
        data = json.load(input_file)
    for index, key in enumerate(data.keys()):
        print('Process {}th file: {}'.format(index, key))
        video_data = data[key]
        output_path = os.path.join(caption_folder, '{}.json'.format(key))
        with open(output_path, 'w') as write_file:
            json.dump(video_data, write_file)

        # print(key, data[key])
        #assert False
    # print(data.keys())
    # pass
