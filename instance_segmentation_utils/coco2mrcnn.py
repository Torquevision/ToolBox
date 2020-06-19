from PIL import Image
import glob
from pycocotools.coco import COCO
import json
import os
import cv2
import argparse

def save_via_region( src_dir, img_dir, count, file_objs, output_dir='/content/dataset_v2'):
    src_dir = src_dir + '/*.json'
    files = glob.glob(src_dir)

    for file in files:
        coco_instance = COCO(file)
        coco_imgs = coco_instance.imgs
        for i, img in enumerate(coco_imgs):
            voc_ann = {}
            im_path = ''
            if '/' in coco_imgs[img]['file_name']:
                im_path = coco_imgs[img]['file_name'].split('/')[-1]
            else:
                im_path = coco_imgs[img]['file_name'].split('\\')[-1]
            # print('==>', im_path)
            image = cv2.imread(os.path.join(img_dir, im_path))
            if image is None:
                print('*'*6, f'{os.path.join(img_dir, im_path)} not found', '*'*6)
            im_path = str(count) + '.jpg'
            cv2.imwrite( os.path.join( output_dir,im_path),image)
            voc_ann['filename'] = str(count) + '.jpg'
            voc_ann['width'] = coco_imgs[img]['width']
            voc_ann['height'] = coco_imgs[img]['height']
            voc_ann['regions'] = {}
            anns_ids = coco_instance.getAnnIds(img)
            anns = coco_instance.loadAnns(anns_ids)
            if not anns:
                continue
            i = 0
            for ann in anns:
                voc_ann['regions'][str(i)] = {}
                voc_ann['regions'][str(i)]['region_attributes'] = {"object_name": str(ann['category_id'])}
                xs, ys = [], []
                for j in range(len(ann['segmentation'][0])):
                    if j%2 == 0:
                        xs.append(int(ann['segmentation'][0][j]))
                    else:
                        ys.append(int(ann['segmentation'][0][j]))
                voc_ann['regions'][str(i)]['shape_attributes'] = {'all_points_x': xs, 'all_points_y': ys, "name": "polygon"}
                i += 1
            # print(voc_ann)
            file_objs.append(voc_ann)
            count += 1
        #     break
        # break
    # json.dump(file_objs, open(os.path.join(output_dir, 'via_region_data.json'), 'w'))
    return count, file_objs

def parse_input():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dir', default=None, help='directory contains images and annotations.')
    parser.add_argument('--dirs', default=None, help='directories contains images and annotations.')
    parser.add_argument('--target', default=None, help='Output folder.')
    return parser.parse_args()


def main():
    args = parse_input()
    
    assert args.dir is not None or args.dirs is not None, 'Either dir or dirs must be provided in the input.'
    
    assert args.target is not None, '--target folder must be provided'
    
    if not os.path.exists(args.target):
        os.mkdir(args.target)
    
    dirs = None
    if args.dir:
        dirs = [args.dir]
    else:
        dirs = glob.glob(args.dirs + '/*')
    
    target = args.target
    count = 0
    file_objs = []
    for dir in dirs:
        if not os.path.isdir(dir):
            print(f'--> Skipping {dir} because it is not a folder')
            continue
        print( '\n--> processing', dir)
        ann_dir = os.path.join(dir, 'annotations')
        imgs_dir = os.path.join(dir, 'images')
        count, file_objs = save_via_region( ann_dir, imgs_dir, count, file_objs, output_dir=target)
    # print(len(file_objs))
    json.dump(file_objs, open(os.path.join(target, 'via_region_data.json'), 'w'))

    print('--> files exported at', target)


if __name__ == '__main__':
    main()