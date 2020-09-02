import os

if __name__ == '__main__':
    feat_folder = '/home/liangkeg/main_storage/data/howto100m/features/2d'
    feat_files = os.listdir(feat_folder)
    with open('server_1.txt', 'w') as fw:
        for feat_file in feat_files:
            if feat_file.endswith('.npy'):
                fw.write('{}\n'.format(feat_file))
