
# Requirement: install youtube-dl (https://github.com/rg3/youtube-dl/)

import os
import threading

max_th = 8
dataset_root = '/home/liangkeg/howto100m_dataset/raw_videos'
if not os.path.isdir(dataset_root):
    os.makedirs(dataset_root)
def download_video(vid_url):
    cmd = 'wget --user liangke --password ffe9664e5da094d {} -P {}'.format(vid_url, dataset_root)
    os.system(cmd)
# dataset_root = '../raw_videos'
# dataset_root = '/home/liangkeg/howto100m_dataset/raw_videos'
# if not os.path.isdir(dataset_root):
#     os.makedirs(dataset_root)

missing_vid_lst = []

vid_files = set()
with open('./download_links.txt', 'r') as fr:
    for line in fr:
        vid_files.add(line.rstrip('\r\n'))
thread_pool=[]

for vid_url in vid_files:
    while len(threading.enumerate()) >= max_th:
        pass
    now_th = threading.Thread(target=download_video, args=[vid_url])
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
os.system("find /home/liangkeg/howto100m_dataset/raw_videos -name '*.f*' -delete")
