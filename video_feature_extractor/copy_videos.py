import os
import threading

def copy_video(src_path, dst_path):
    cmd = 'scp {} {}'.format(src_path, dst_path)
    print(cmd)
    os.system(cmd)


if __name__ == '__main__':

    file_path = './all_second.csv'
    dst_root = '/home/liangkeg/second_howto100m_dataset/raw_videos'
    max_th = 8
    thread_pool = []

    with open(file_path, 'r') as fr:
        for index, line in enumerate(fr):
            if index == 0:
                continue
            line = line.rstrip('\r\n')
            src_path = line.split(',')[0]
            dst_path = line.split(',')[1]
            # video_name = os.path.basename(src_path)
            # dst_path = os.path.join(dst_root, video_name)
            if os.path.exists(dst_path):
                continue
            while len(threading.enumerate()) >= max_th:
                pass
            now_th = threading.Thread(target=copy_video, args=[src_path, dst_path])
            now_th.start()
            thread_pool.append(now_th)
        for th in thread_pool:
            th.join()


