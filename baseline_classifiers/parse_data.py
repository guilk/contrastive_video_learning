import os
import json
import math

def load_data(file_path):
    with open(file_path, "r") as read_file:
        data = json.load(read_file)
    return data

def count_num(folder_path):
    files = os.listdir(folder_path)
    imgs = [img for img in files if img.endswith('.jpg') and (not img.startswith('._'))]
    return len(imgs)

def get_start_end_frame(annotations, duration, folder_path):
    start_time, end_time = annotations['segment']
    total_num = count_num(folder_path)
    start_frame = int(math.floor(start_time/duration * total_num))
    end_frame = int(math.ceil(end_time/duration * total_num))
    return start_frame, end_frame


def copy_frames(folder_path, start_frame, end_frame):
    dst_folder = '/home/liangkeg/third_hand/tmp'
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    for index in range(start_frame, end_frame+1, 1):
        img_name = '{}.jpg'.format(str(index).zfill(6))
        img_path = os.path.join(folder_path, img_name)
        cmd = 'scp {} {}'.format(img_path, os.path.join(dst_folder, img_name))
        os.system(cmd)


if __name__ == '__main__':
    ann_path = '/home/liangkeg/gpu4ssd/third_hand/YouCookII/annotations/youcookii_annotations_trainval.json'
    frames_folder = '/home/liangkeg/gpu4ssd/third_hand/YouCookII/raw_frames'
    anns = load_data(ann_path)
    video_infos = anns['database']
    counter = 0

    fw = open('./youcook_clip_info.txt', 'w')

    for video_name in video_infos:
        video_info = video_infos[video_name]
        duration = video_info['duration']
        subset = video_info['subset']
        folder_path = os.path.join(frames_folder, video_name)
        if not os.path.exists(folder_path):
            counter += 1
            continue
        # print(video_name)
        for annotations in video_info['annotations']:
            start_time, end_time = annotations['segment']
            sentence = annotations['sentence']
            id = annotations['id']
            start_frame, end_frame = get_start_end_frame(annotations, duration, folder_path)
            line = '{},{},{},{},{},{},{}\n'.format(video_name, start_frame, end_frame, sentence, start_time, end_time, subset)
            print(line)
            fw.write(line)
    fw.close()
    print('There are {} videos not found'.format(counter))