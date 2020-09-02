import os
import json


if __name__ == '__main__':
    file_path = './00Lty3r6JLE.json'
    with open(file_path, 'r') as input_file:
        data = json.load(input_file)

    start_times = data['start']
    end_times = data['end']
    text_lines = data['text']
    num_clips = len(start_times)

    for index in range(num_clips-1):
        # cur_line, next_line = text_lines[index], text_lines[index+1]
        # print('{} - {}'.format(cur_line, next_line))
        start_time, end_time, text_line = start_times[index], end_times[index], text_lines[index]
        # for start_time, end_time, text_line in zip(start_times, end_times, text_lines):
        print(start_time, end_time, text_line)