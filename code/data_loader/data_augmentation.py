import numpy as np
import pandas as pd
import os
import sys
import matplotlib.image as mpimg
from joblib import Parallel, delayed
import multiprocessing
import time


from PIL import Image
import png


def processInput(image_name, train_labels, filter_list, num_augs, data_folder):
    aug_data_folder = os.path.join(data_folder, 'train_aug_par')
    for image_name in train_labels.Id:
        image_target = train_labels[train_labels.Id == image_name].Target.values[0]
        for colour in filter_list:
            image_path = os.path.join(data_folder, 'train', image_name + '_' + colour + '.png')

            max_num_rot = min(num_augs[label_names[int(num)]] for num in image_target.split(' '))

            # Augmenting the image set
            for i_rot in range(0, max_num_rot):
                # Rotating the image
                rot_image = np.rot90(mpimg.imread(image_path), i_rot+1)
                # Convert array to image
                img = Image.fromarray(rot_image * 255)  # Multiply by 255 because original values from 0 to 1
                # img.show()  # Uncomment to view image
                # Save newly created images
                # Convert image to RBG mode (because original - possible CMYK - not supported by PNG)
                img.convert('RGB').save(
                    os.path.join(aug_data_folder, image_name + '_rot' + str(i_rot+1) + '_' + colour + '.png'))

                # Same for reversed image
                rev_image = np.fliplr(np.rot90(mpimg.imread(image_path), i_rot+1))
                img = Image.fromarray(rev_image * 255)  # Multiply by 255 because original values from 0 to 1
                # Save image
                img.convert('RGB').save(
                    os.path.join(aug_data_folder, image_name + '_rev' + str(i_rot+1) + '_' + colour + '.png'))

    return {[image_name + '_rot' + str(i_rot+1), image_target], [image_name + '_rev' + str(i_rot+1), image_target]}


def data_aug(data_folder, train_labels, label_names, parallelization_bool=True):
    print('Starting data augmentation')

    train_data_folder = os.path.join(data_folder, 'train')

    # Adding 1 column per target in dataset, and setting value to 1 if it is in image's target label
    train_labels = train_labels.apply(fill_targets, axis=1)

    # Computing number of augmentations needed for each image
    num_augs = num_aug_perlabel(train_labels)

    filter_list = ['yellow', 'red', 'blue', 'green']

    # Parallelizing process
    if parallelization_bool:
        print("Parallelizing...")

        # Create data/train_aug folder if it does not exist yet
        if not os.path.exists(os.path.join(data_folder, 'train_aug_par')):
            print('Creating train_aug _par data folder')
            os.makedirs(os.path.join(data_folder, 'train_aug_par'))

        num_cores = multiprocessing.cpu_count()

        rebalanced_images = Parallel(n_jobs=num_cores)(delayed(processInput)(image_name, train_labels, filter_list,
                                                                             num_augs, data_folder)
                                                       for image_name in train_labels.Id)
        save_obj(rebalanced_images, os.path.join(data_folder, 'augmented_train_par.csv'))



    # If no parallelization
    else:

        # Create data/train_aug folder if it does not exist yet
        if not os.path.exists(os.path.join(data_folder, 'train_aug')):
            print('Creating train_aug data folder')
            os.makedirs(os.path.join(data_folder, 'train_aug'))

        aug_data_folder = os.path.join(data_folder, 'train_aug')
        rebalanced_images = []

        counter = 0
        t_start = time.time()

        for image_name in train_labels.Id:
            image_target = train_labels[train_labels.Id == image_name].Target.values[0]

            # Get minimum number of rotations/reversions needed (use value for most common target label)
            max_num_rot = min(num_augs[label_names[int(num)]] for num in image_target.split(' '))

            for i_rot in range(0, int(max_num_rot / 2)):
                # Augmenting the image set
                for colour in filter_list:
                    image_path = os.path.join(data_folder, 'train', image_name + '_' + colour + '.png')
                    # Rotating the image
                    rot_image = np.rot90(mpimg.imread(image_path), i_rot+1)
                    # Convert array to image
                    img = Image.fromarray(rot_image * 255)  # Multiply by 255 because original values from 0 to 1
                    # img.show()  # Uncomment to view image
                    # Save newly created images
                    # Convert image to RBG mode (because original - possible CMYK - not supported by PNG)
                    img.convert('RGB').save(
                        os.path.join(aug_data_folder, image_name + '_rot' + str(i_rot+1) + '_' + colour + '.png'))

                    # Same for reversed image
                    rev_image = np.fliplr(np.rot90(mpimg.imread(image_path), i_rot+1))
                    img = Image.fromarray(rev_image * 255)  # Multiply by 255 because original values from 0 to 1
                    # Save image
                    img.convert('RGB').save(
                        os.path.join(aug_data_folder, image_name + '_rev' + str(i_rot+1) + '_' + colour + '.png'))

                rebalanced_images.append([image_name + '_rot' + str(i_rot+1), image_target])
                rebalanced_images.append([image_name + '_rev' + str(i_rot+1), image_target])

            if counter % 100 == 0:
                print('Processed {} images out of {}'.format(counter, len(train_labels)))
            if counter % 500 == 0:
                print("{}s. elapsed".format(time.time() - t_start))
            counter += 1

    t_end = time.time()
    print("Data augmentation took {}s.".format(t_end - t_start))

    rebalanced_images = pd.DataFrame(rebalanced_images, columns=['Id', 'Target'])
    # Save dataframe
    rebalanced_images.to_csv(os.path.join(data_folder, 'augmented_train.csv'))
    print("Saved dataframe as augmented_train.csv in data folder")

    return rebalanced_images

def num_aug_perlabel(train_labels):
    # Find number of augmentations necessary
    sorted_label_values = train_labels.drop(["Id", "Target"], axis=1).sum(axis=0).sort_values(ascending=False)
    print(sorted_label_values)
    aug = 0
    num_augs = {sorted_label_values.index[0]: 0}

    for i in range(1, len(sorted_label_values )):
        while (sorted_label_values[0] / sorted_label_values[i] > aug) & (aug < 8) & (
                sorted_label_values[i] * 2 < sorted_label_values[0]):
            aug += 2
        num_augs[sorted_label_values.index[i]] = aug
    return num_augs

#Function to get target count for each image
def fill_targets(row):
    for key in row.Target.split(" "):
        row.loc[label_names[int(key)]] = 1
    return row

def save_obj(obj, name):
    """
    Shortcut function to save an object as pkl
    Args:
        obj: object to save
        name: filename of the object
    """
    with open(name + '.pkl', 'wb') as f:
        pkl.dump(obj, f, pkl.HIGHEST_PROTOCOL)



if __name__=='__main__':
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


    label_names = {
        0: "Nucleoplasm",
        1: "Nuclear membrane",
        2: "Nucleoli",
        3: "Nucleoli fibrillar center",
        4: "Nuclear speckles",
        5: "Nuclear bodies",
        6: "Endoplasmic reticulum",
        7: "Golgi apparatus",
        8: "Peroxisomes",
        9: "Endosomes",
        10: "Lysosomes",
        11: "Intermediate filaments",
        12: "Actin filaments",
        13: "Focal adhesion sites",
        14: "Microtubules",
        15: "Microtubule ends",
        16: "Cytokinetic bridge",
        17: "Mitotic spindle",
        18: "Microtubule organizing center",
        19: "Centrosome",
        20: "Lipid droplets",
        21: "Plasma membrane",
        22: "Cell junctions",
        23: "Mitochondria",
        24: "Aggresome",
        25: "Cytosol",
        26: "Cytoplasmic bodies",
        27: "Rods & rings"
    }

    parallelization_bool = sys.argv[1]
    data_aug(cwd, tmp, label_names, parallelization_bool)
