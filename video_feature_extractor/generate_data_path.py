import os

if __name__ == '__main__':
    dataset_name = 'howto100m'
    feat_type = '2d'
    head_line = 'video_path,feature_path'

    fw = open('./{}_{}.csv'.format(dataset_name, feat_type), 'w')
    fw.write('{}\n'.format(head_line))
    dataset_root = '/home/liangkeg/howto100m_dataset/raw_videos'
    videos = os.listdir(dataset_root)
    video_dict = {}
    for video in videos:
        video_dict[video.split('.')[0]] = video

    cand_list_file = '/home/liangkeg/main_storage/data/metadata/HowTo100M_v1.csv'
    feat_root = '/home/liangkeg/main_storage/data/howto100m/features/{}'.format(feat_type)
    if not os.path.exists(feat_root):
        os.makedirs(feat_root)

    with open(cand_list_file, 'r') as fr:
        for line in fr:
            video_name = line.split(',')[0]
            if video_name in video_dict:
                src_file = os.path.join(dataset_root, video_dict[video_name])
                dst_file = os.path.join(feat_root, '{}.npy'.format(video_name))
                if not os.path.exists(dst_file):
                    dst_file = '{}.npy'.format(video_name)
                    fw.write('{},{}\n'.format(src_file, dst_file))
    fw.close()
