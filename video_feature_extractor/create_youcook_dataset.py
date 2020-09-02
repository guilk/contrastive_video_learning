import os


if __name__ == '__main__':
    video_root = '/home/liangkeg/main_storage/data/youcookii/raw_videos'
    feat_root = '/home/liangkeg/main_storage/data/youcookii/features/3d'
    file_path = 'youcook_list.csv'

    fw = open(file_path, 'w')
    fw.write('video_path,feature_path\n')
    videos = os.listdir(video_root)
    for video_name in videos:
        video_path = os.path.join(video_root, video_name)
        feat_name = '{}.npy'.format(video_name.split('.')[0])
        feat_path = os.path.join(feat_root, feat_name)
        if os.path.exists(feat_path):
            continue
        line = '{},{}\n'.format(video_path, feat_name)
        fw.write(line)
    fw.close()