
# Requirement: install youtube-dl (https://github.com/rg3/youtube-dl/)

import os

import threading

max_th = 8

def download_video(vid_prefix, vid_url):
    cmd = ' '.join(("youtube-dl -o", vid_prefix, vid_url))
    print(cmd)
    # os.system(cmd)
# dataset_root = '../raw_videos'
dataset_root = '/home/liangkeg/main_storage/data/youcookii/raw_videos'
if not os.path.isdir(dataset_root):
    os.makedirs(dataset_root)

missing_vid_lst = []


vid_files = set()
with open('./youcook_clip_info.txt', 'r') as fr:
    for line in fr:
        vid_files.add(line.split(',')[0])

thread_pool=[]

for vid_name in vid_files:
    vid_url = 'www.youtube.com/watch?v=' + vid_name
    vid_prefix = os.path.join(dataset_root, vid_name)

    while len(threading.enumerate()) >= max_th:
        pass
    now_th = threading.Thread(target=download_video, args=[vid_prefix, vid_url])
    now_th.start()
    thread_pool.append(now_th)

for th in thread_pool:
    th.join()

# write the missing videos to file
missing_vid = open('missing_videos.txt', 'w')
for line in missing_vid_lst:
    missing_vid.write(line)

# sanitize and remove the intermediate files
# os.system("find ../raw_videos -name '*.part*' -delete")
os.system("find /home/liangkeg/main_storage/data/youcookii/raw_videos -name '*.f*' -delete")
