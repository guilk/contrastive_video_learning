import os

def to_process_videos():
    num_gpus = 8
    videos_list = []
    for gpu_index in range(num_gpus):
        file_name = 'list_1_{}.csv'.format(gpu_index)
        print(file_name)
        with open('./{}'.format(file_name), 'r') as fr:
            for index, line in enumerate(fr):
                if index == 0:
                    continue
                video_path = line.split(',')[0]
                videos_list.append(video_path)

    return videos_list


if __name__ == '__main__':
    dst_root = '/home/liangkeg/second_howto100m_dataset/raw_videos'
    videos_list = to_process_videos()
    print(len(videos_list), len(set(videos_list)))
    file_path = './all_second.csv'
    with open(file_path, 'w') as fw:
        fw.write('src_path,dst_path\n')
        for video_path in videos_list:
            video_name = os.path.basename(video_path)
            fw.write('{},{}\n'.format(video_path, os.path.join(dst_root, video_name)))
