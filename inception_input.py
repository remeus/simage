import os

import numpy as np
from scipy import misc

import inception



def cache_images_dir(dir, max_n=1000, replace=False):

    # Inception
    inception.maybe_download()
    model = inception.Inception()

    # Storage
    file_path_cache = os.path.join(dir + '/images_features.pkl')

    if replace:
        os.remove(file_path_cache)

    print("Processing Inception transfer-values...")

    dir_im = dir + '/pics'

    n_total_images = sum([len(files) for r, d, files in os.walk(dir_im)])
    n_total_images = min([max_n, n_total_images])
    print('Fetching %d images in %s ...' % (n_total_images, dir_im))
    images = np.zeros((n_total_images, 192, 256, 3), dtype=np.float32)
    id = []
    index = 0
    n_err = 0
    for d in os.listdir(dir_im):
        if index >= max_n:
            break
        d = dir_im + '/' + d
        for image_name in os.listdir(d):
            if index >= max_n:
                break
            image_path = d + '/' + image_name
            try:
                image_data = (misc.imread(image_path)[:, :, :3]).astype(np.float32)
                images[index, :, :, :] = image_data  # (n, height, width, channels)
                id.append(os.path.splitext(image_name)[0])
                index += 1
            except OSError as err:
                print(err)
                n_err += 1
    if n_err > 0:
        images = np.delete(images, range(n_total_images - n_err, n_total_images), 0)

    id = np.array(id)

    transfer_values = inception.transfer_values_cache(cache_path=file_path_cache, images=images, model=model)

    return transfer_values, id