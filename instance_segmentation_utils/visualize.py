
import json
import os
import glob
import random
import shutil
import argparse

import numpy as np
import cv2

def view_dataset(dataset_path):

    json_file = 'via_region_data.json'

    json_data = json.load(open(os.path.join(dataset_path, json_file)))

    for file in json_data:
        if file['filename'].endswith('xml'):
            file['filename'] = file['filename'][:-3] + 'jpg'
        print(file['filename'])
        img_path = os.path.join( dataset_path, file['filename'])
        img = cv2.imread(img_path)
        frame = img.copy()
        if img is None:
            print(img_path)
            continue

        for region in file['regions'].values():
            polygon = []
            for x, y in zip(region['shape_attributes']['all_points_x'], region['shape_attributes']['all_points_y']):
                polygon.append([x, y])
            polygon = np.array(polygon)
            img = cv2.fillPoly(img, pts =[polygon], color=(255,0,0))
        cv2.imshow('img', img)
        cv2.imshow('frame', frame)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break

def merge_dataset(dirs_path='', target_dir=''):
    dirs = glob.glob(dirs_path + '/*')
    ref_file = {}
    new_objs = []
    ind = 0
    new_count = 100000
    for dir in dirs:
        print(dir)
        file = os.path.join(dir, 'via_region_data.json')
        data = json.load(open(file, 'r'))
        print(len(data))
        for obj in data:
            if obj['filename'] in ref_file:
                img_path = os.path.join(dir, obj['filename'])
                obj['filename'] = str(new_count) + '.jpg'
                new_count += 1
            else:
                img_path = os.path.join(dir, obj['filename'])
            img = cv2.imread(img_path)
            cv2.imwrite(os.path.join(target_dir, obj['filename']), img)
            print(f'new file created at {img_path}')
            ref_file[obj['filename']] = True
            new_objs.append(obj)
            ind += 1
            # break
    print( new_objs[0], len(new_objs))
    random.shuffle(new_objs)
    train_ = new_objs[:500]
    val_ = new_objs[500:]
    for v in val_:
        shutil.copy(os.path.join(target_dir, v['filename']), os.path.join(target_dir, 'val', v['filename']))
    json.dump(train_, open(os.path.join(target_dir, 'via_region_data.json'), 'w'))
    json.dump(val_, open(os.path.join(target_dir, 'val', 'via_region_data.json'), 'w'))

# merge_dataset(dirs_path=r'D:\my-jobs\ice-hockey\Ash_vhockey_proj_vid_1\dataset_main', target_dir=r'D:\my-jobs\ice-hockey\Ash_vhockey_proj_vid_1\data_merged')

def parse_input():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dir', default=None, help='directory contains images and annotation file.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_input()
    view_dataset(args.dir)