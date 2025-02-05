import cv2
import numpy as np
import os
from shutil import copyfile
import yaml


def mask_to_polygon(mask):
    # Ensure the mask is grayscale
    if len(mask.shape) == 3:  # if the image has 3 channels (RGB)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = [contour.flatten().tolist() for contour in contours if len(contour) > 2]
    return polygons


def normalize_coordinates(polygons, img_width, img_height):
    normalized_polygons = []
    for polygon in polygons:
        normalized_polygon = []
        for i in range(0, len(polygon), 2):
            normalized_polygon.append(polygon[i] / img_width)  # x
            normalized_polygon.append(polygon[i + 1] / img_height)  # y
        normalized_polygons.append(normalized_polygon)
    return normalized_polygons


def create_yolo_annotation(labels_file, class_id, polygons):
    annotations = []
    for polygon in polygons:
        polygon_str = " ".join(map(str, polygon))
        annotations.append(f"{class_id} {polygon_str}")
    
    # Save to file
    with open(labels_file, 'w') as f:
        f.write("\n".join(annotations))


if __name__ == '__main__':

    # TODO: also create test set

    train_val_test_split = [0.6, 0.2, 0.2]

    assert sum(train_val_test_split) == 1.0, f'Sum of the train val test split has to be 1.0, but is {sum(train_val_test_split)}'

    mask_folder = 'mvtec_anomaly_detection/hazelnut/ground_truth'
    img_folder = 'mvtec_anomaly_detection/hazelnut/test'

    out_folder = 'dataset'

    os.makedirs(out_folder, exist_ok=True)
    os.makedirs(os.path.join(out_folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(out_folder, 'images/train'), exist_ok=True)
    os.makedirs(os.path.join(out_folder, 'images/val'), exist_ok=True)
    os.makedirs(os.path.join(out_folder, 'images/test'), exist_ok=True)
    os.makedirs(os.path.join(out_folder, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(out_folder, 'labels/train'), exist_ok=True)
    os.makedirs(os.path.join(out_folder, 'labels/val'), exist_ok=True)
    os.makedirs(os.path.join(out_folder, 'labels/test'), exist_ok=True)

    dataset_config = {
        'path': os.path.join(os.getcwd(), out_folder),
        'train': 'images/train',
        'val': 'images/val',
        'names': {},
    }

    for class_id, category in enumerate(os.listdir(img_folder)):
        print(category, class_id)

        dataset_config['names'][class_id] = category

        # TODO: randomize images

        number_of_images = len(os.listdir(os.path.join(img_folder, category)))

        for index, img_name in enumerate(os.listdir(os.path.join(img_folder, category))):

            if index <= number_of_images * train_split:
                mode = 'train'
            else:
                mode = 'val'

            img_fn = os.path.join(img_folder, category, img_name)
            mask_fn = os.path.join(mask_folder, category, img_name[:-4] + '_mask.png')

            labels_file = os.path.join(out_folder, 'labels', mode, f'{class_id}_{img_name[:-4]}.txt')

            if os.path.exists(mask_fn):

                mask_img = cv2.imread(mask_fn)
                height, width, _ = np.shape(mask_img)

                polygons = mask_to_polygon(mask_img)

                normalized_polygons = normalize_coordinates(polygons, width, height)

                create_yolo_annotation(labels_file, class_id, normalized_polygons)
            
            else:
                with open(labels_file, 'w') as f:
                    f.write('')

            copyfile(img_fn, os.path.join(out_folder, 'images', mode, f'{class_id}_{img_name}'))


    with open(rf'{out_folder}/dataset.yaml', 'w') as f:
        yaml.dump(dataset_config, f)

