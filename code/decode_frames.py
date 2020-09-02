import os

import threading

def decode_video(dst_root, src_path):
    if os.path.exists(dst_root):
        return
    else:
        os.makedirs(dst_root)
    cmd = 'ffmpeg -i {} {}/%6d.jpg'.format(src_path, dst_root)
    os.system(cmd)

if __name__ == '__main__':
    video_root = '/home/liangkeg/main_storage/data/youcookii/raw_videos'
    frame_root = '/home/liangkeg/main_storage/data/youcookii/raw_frames'
    max_th = 8
    thread_pool = []
    videos = os.listdir(video_root)
    for video_name in videos:
        dst_root = os.path.join(frame_root, video_name.split('.')[0])
        src_path = os.path.join(video_root, video_name)
        while len(threading.enumerate()) >= max_th:
            pass
        now_th = threading.Thread(target=decode_video, args=[dst_root, src_path])
        now_th.start()
        thread_pool.append(now_th)
    for th in thread_pool:
        th.join()

    #
    # video_splits = ['training', 'validation', 'testing']
    # video_types = ['.mkv', '.mp4', '.avi']
    # # counter = 0
    # for video_split in video_splits:
    #     video_folder = os.path.join(video_root, video_split)
    #     folders = os.listdir(video_folder)
    #     for folder in folders:
    #         folder_path = os.path.join(video_folder, folder)
    #         video_names = os.listdir(folder_path)
    #         for video_name in video_names:
    #             dst_root = os.path.join(frame_root, video_name.split('.')[0])
    #             src_path = os.path.join(folder_path, video_name)
    #             if not os.path.exists(dst_root):
    #                 # print(dst_root)
    #                 os.makedirs(dst_root)
    #             cmd = 'ffmpeg -i {} {}/%6d.jpg'.format(src_path, dst_root)
    #             os.system(cmd)
    #
    # # print(counter)



    # video_root = '/home/liangkeg/gpu4ssd/third_hand/YouCookII/raw_videos/'
    # frame_root = '/home/liangkeg/gpu4ssd/third_hand/YouCookII/raw_frames'
    #
    # video_splits = ['training', 'validation', 'testing']
    # video_types = ['.mkv', '.mp4', '.avi']
    # # counter = 0
    # for video_split in video_splits:
    #     video_folder = os.path.join(video_root, video_split)
    #     folders = os.listdir(video_folder)
    #     for folder in folders:
    #         folder_path = os.path.join(video_folder, folder)
    #         video_names = os.listdir(folder_path)
    #         for video_name in video_names:
    #             dst_root = os.path.join(frame_root, video_name.split('.')[0])
    #             src_path = os.path.join(folder_path, video_name)
    #             if not os.path.exists(dst_root):
    #                 # print(dst_root)
    #                 os.makedirs(dst_root)
    #             cmd = 'ffmpeg -i {} {}/%6d.jpg'.format(src_path, dst_root)
    #             os.system(cmd)
    #
    # # print(counter)
