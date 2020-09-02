import os


if __name__ == '__main__':
    video_root = '/home/liangkeg/howto100m_data/raw_videos'
    videos = os.listdir(video_root)

    extensions = set()
    for video_name in videos:
        extensions.add(video_name.split('.')[-1])
    print(extensions)
