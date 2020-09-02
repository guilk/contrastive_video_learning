import os
import threading

max_th = 16


def trim_video(clip_root, video_root, line):
    base_name, start_time, end_time, caption = line.rstrip('\r\n').split(',')
    video_name = check_available(video_root, base_name)
    if video_name == '':
        return
    print(line)
    input_path = os.path.join(video_root, video_name)
    output_path = os.path.join(clip_root, video_name)
    cmd = "ffmpeg -i " + input_path + " -ss  " + start_time + " -to " + end_time + " -c copy " + output_path
    os.system(cmd)

def check_available(video_root, base_name):
    file_extns = ['mp4', 'webm']
    video_name = ''
    for file_extn in file_extns:
        video_name = '{}.{}'.format(base_name, file_extn)
        video_path = os.path.join(video_root, video_name)
        print(video_path)
        if os.path.exists(video_path):
            return video_name
        else:
            video_name = ''
    return video_name

if __name__ == '__main__':

    video_root = '/home/liangkeg/howto100m_data/raw_videos'
    clip_root = '/home/liangkeg/howto100m_data/clips'
    thread_pool = []
    max_th = 16
    with open('../candidate_clips.txt', 'r') as fr:
        for line in fr:
            while len(threading.enumerate()) >= max_th:
                pass
            now_th = threading.Thread(target=trim_video, args=[clip_root, video_root, line])
            now_th.start()
            thread_pool.append(now_th)
    for th in thread_pool:
        th.join()



