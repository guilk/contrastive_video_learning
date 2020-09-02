import os


if __name__ == '__main__':

    completed_videos = set()
    with open('completed_videos.txt', 'r') as fr:
        for line in fr:
            completed_videos.add(line.split('.')[0])


    to_process_videos = set()
    src_file = '/home/liangkeg/main_storage/data/metadata/HowTo100M_v1.csv'
    with open(src_file, 'r') as fr:
        for index, line in enumerate(fr):
            if index == 0:
                continue
            video_name = line.split(',')[0]
            if video_name not in completed_videos:
                to_process_videos.add(video_name)

    dataset_root = '/home/liangkeg/howto100m_dataset/raw_videos'
    videos = os.listdir(dataset_root)
    video_dict = {}
    for video in videos:
        video_dict[video.split('.')[0]] = video
    
    to_process_videos = list(to_process_videos)
    num_servers = 2
    num_gpus = 8
    feat_type = '2d'
    feat_root = '/home/liangkeg/main_storage/data/howto100m/features/{}'.format(feat_type)
    list_len = int(len(to_process_videos)/(num_servers * num_gpus))
    for server_index in range(num_servers):
        for gpu_index in range(num_gpus):
            start_index = (server_index * num_gpus + gpu_index) * list_len
            end_index = start_index + list_len
            videos = to_process_videos[start_index:end_index]
            with open('list_{}_{}.csv'.format(server_index, gpu_index), 'w') as fw:
                fw.write('video_path,feature_path\n')
                for video_name in videos:
                    if video_name not in video_dict:
                        continue
                    src_file = os.path.join(dataset_root, video_dict[video_name])
                    dst_file = '{}.npy'.format(video_name)
                    fw.write('{},{}\n'.format(src_file, dst_file))
