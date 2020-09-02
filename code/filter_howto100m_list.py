import os



if __name__ == '__main__':
    dataset_root = '/home/liangkeg/howto100m_dataset/raw_videos'
    videos = os.listdir(dataset_root)
    downloaded_videos = [video.split('.')[0] for video in videos]


    src_list_file = './howto100m_videos.txt'
    link_dict = {}
    with open('./howto100m_videos.txt', 'r') as fr:
        for line in fr:
            video = os.path.basename(line)
            video_name = video.split('.')[0]
            link_dict[video_name] = line

    fw = open('download_links.txt', 'w')
    cand_list_file = '/home/liangkeg/main_storage/data/metadata/HowTo100M_v1.csv'
    with open(cand_list_file, 'r') as fr:
        for line in fr:
            video_name = line.split(',')[0]
            if video_name not in downloaded_videos:
                if video_name in link_dict:
                    # print(link_dict[video_name])
                    fw.write(link_dict[video_name])

    fw.close()
