import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
#sns.set()

from PIL import Image
import png


def data_aug(data_folder, train_labels):

    # Create data/train_aug folder if it does not exist yet
    if not os.path.exists(os.path.join(data_folder, 'train_aug')):
        os.makedirs(os.path.join(data_folder, 'train_aug'))

    aug_data_folder = os.path.join(data_folder, 'train_aug')

    filter_list = ['yellow', 'red', 'blue', 'green']
    rebalanced_images = []

    for image_name in imbalanced_train_labels.Id:
        image_target = imbalanced_train_labels[imbalanced_train_labels.Id == image_name].Target.values[0]
        for colour in filter_list:
            image_path = folder + image + '_' + colour + '.png'
            for i_rot in range(1, 4):
                rot_image = np.rot90(mpimg.imread(image_path), i_rot)
                # Convert array to image
                img = Image.fromarray(rot_image * 255)  # Multiply by 255 because original values from 0 to 1
                # img.show()  # Uncomment to view image
                # Save newly created images
                # Convert image to RBG mode (because original - possible CMYK - not supported by PNG)
                img.convert('RGB').save(
                    os.path.join(aug_data_folder, image_name + '_rot' + str(i_rot) + '_' + colour + '.png'))

                # Same for reversed image
                rev_image = np.fliplr(np.rot90(mpimg.imread(image_path), i_rot))
                img = Image.fromarray(rev_image * 255)  # Multiply by 255 because original values from 0 to 1
                img.convert('RGB').save(
                    os.path.join(aug_data_folder, image_name + '_rev' + str(i_rot) + '_' + colour + '.png'))

                rebalanced_images.append([image_name + '_rot' + str(i_rot), image_target])
                rebalanced_images.append([image_name + '_rev' + str(i_rot), image_target])

def num_aug():
    # Find number of augmentations necessary
    sorted_label_values = train_labels.drop(["Id", "Target"], axis=1).sum(axis=0).sort_values(ascending=False)
    aug = 0
    num_augs = {sorted_label_values.index[0]: 0}

    for i in range(1, len(sorted_label_values )):
        while (sorted_label_values[0] / sorted_label_values[i] > aug) & (aug < 8) & (
                sorted_label_values[i] * 2 < sorted_label_values[0]):
            aug += 2
        num_augs[sorted_label_values.index[i]] = aug
    return num_augs


if __name__=='__main__':
    data_folder = 'data'
    cwd = os.getenv("DATA_PATH")
    if cwd is None:
        print("Set your DATA_PATH env first")
        sys.exit(1)
    # Read csv file
    tmp = pd.read_csv(os.path.abspath(cwd + 'train.csv'),
                      delimiter=',', engine='python')
    # A vector of images id.
    image_ids = tmp["Id"]
    n = len(image_ids)
    # for each id sublist of the 4 filenames [batch_size, 4]
    filenames = np.asarray([[
        cwd + '/train/' + id + '_' + c + '.png'
        for c in ['red', 'green', 'yellow', 'blue']
    ] for id in image_ids])
    # Labels
    labels = tmp["Target"].values
    train_labels = pd.read_csv('data/train.csv', sep=',')

    data_aug(cwd, tmp)
