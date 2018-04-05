from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import sys
import time

import numpy as np
from sklearn.decomposition import TruncatedSVD
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import TSNE


def load_images(dir, max_n_images=sys.maxsize):
    dir = dir + '/pics'
    n_total_images = sum([len(files) for r, d, files in os.walk(dir)])
    n_total_images = min([n_total_images, max_n_images])
    print('Fetching %d ids in %s ...' % (n_total_images, dir))
    id = []
    index = 0
    for d in os.listdir(dir):
        if index >= max_n_images:
            break
        d = dir + '/' + d
        for image_name in os.listdir(d):
            if index >= max_n_images:
                break
            try:
                id.append(os.path.splitext(image_name)[0])
                index += 1
            except OSError as err:
                print(err)
    id = np.array(id)
    return id # Images and ids in the same order


def get_training_lookup_table(train_dir):
    link = train_dir + '/pickle/combined.pickle'
    with open(link, 'rb') as f:
        all_txt = pickle.load(f)
    whole_labels = []
    for _, value in all_txt.items():
        whole_labels.append(value)
    ref_labels = [y[0] for x in whole_labels for y in x]
    ref_labels = list(set(ref_labels))
    lookup_table = dict((x, y) for (x, y) in zip(ref_labels, range(len(ref_labels))))  # Match label / number
    return lookup_table # Order unknown


def create_one_hot_label_vector(ids, dir, lookup_table):
    link = dir + '/pickle/combined.pickle'
    with open(link, 'rb') as f:
        all_txt = pickle.load(f)
    res = []
    for id in ids:
        res_i = np.zeros(len(lookup_table), dtype=np.float32)
        try:
            labels_item = all_txt[id] # List of (label, confidence)
            for l in labels_item:  # Tuple (label, confidence)
                try:
                    res_i[lookup_table[l[0]]] = l[1]  # One-hot vector
                except KeyError:
                    print('%s is not in the training set, it has been discarded.' % l[0])
            res.append(res_i)
        except KeyError:
            print('%s is not in the pickle file' % id)
    res = np.array(res)
    return res # Order matches order of ids and images in data


def create_dataset(train_dir,
                   sim_dir,
                   max_n_images,
                   k_retrieved=10,
                   threshold_label=0,
                   n_max_labels=0,
                   n_split=1):

  # Result folder
  if not os.path.exists(sim_dir):
    os.makedirs(sim_dir)

  # n_partial = min([10, max_n_images])
  n_partial = max_n_images

  print('Get data...')
  start = time.time()

  # Load training images
  training_ids = load_images(train_dir, max_n_images)

  # Prepare labels
  lookup_table = get_training_lookup_table(train_dir)

  # Get training table
  labels_tr_mat = create_one_hot_label_vector(training_ids, train_dir, lookup_table)

  # Truncate labels
  if threshold_label > 0:
    main_indexes = np.where(np.sum(labels_tr_mat, 0) > threshold_label)[0]
    print("%d labels will be considered" % len(main_indexes))
    labels_tr_mat = labels_tr_mat[:, main_indexes]

  # SVD
  if n_max_labels > 0:
      svd = TruncatedSVD(n_max_labels).fit(labels_tr_mat)
      labels_tr_mat = svd.transform(labels_tr_mat)

  end = time.time()
  print('Save files...')
  with open(sim_dir + '/sim_ids.pickle', 'wb') as f:
      pickle.dump(training_ids, f)
  with open(sim_dir + '/sim_lookup_table.pickle', 'wb') as f:
      pickle.dump(lookup_table, f)
  print('# Done in %.0f sec' % (end - start))

  # Dimension reduction
  print('Perform dimension reduction...')
  start = time.time()
  print('Truncated SVD...')
  rep = TruncatedSVD(50).fit_transform(labels_tr_mat)
  print('t-SNE...')
  tsne = TSNE(n_components=2)
  rep = tsne.fit_transform(rep)
  end = time.time()
  print('Save file...')
  with open(sim_dir + '/sim_rep.pickle', 'wb') as f:
      pickle.dump(rep, f)
  print('# Done in %.0f sec' % (end - start))

  # Distance mat
  n_images = len(training_ids)
  print('Compute distance vector...')
  start = time.time()
  dist_mat = pdist(rep, 'euclidean')
  end = time.time()
  print('# Done in %.0f sec' % (end - start))
  start = time.time()
  print('Compute distance matrix...')
  dist_mat = squareform(dist_mat)
  end = time.time()
  print('# Done in %.0f sec' % (end - start))

  # Similitude mat
  k = k_retrieved
  assert n_partial % n_split == 0
  spl = n_partial // n_split
  for m in range(n_split):
      print('SPLIT %d / %d' % (m+1, n_split))
      print('Get highest values...')
      start = time.time()
      ind_start = m * spl
      sub_mat = dist_mat[:,ind_start:ind_start+spl]
      top_ind = np.argpartition(-sub_mat, -k, axis=0)[-k:]
      end = time.time()
      print('# Done in %.0f sec' % (end - start))
      print('Compute similitude matrix...')
      start = time.time()
      sim_mat = np.zeros([spl, n_images], dtype=np.bool_)
      for i in range(spl):
          for j in range(k):
              sim_mat[i, top_ind[j, i]] = 1
      print('Save file...')
      if n_split == 1:
        out_file = sim_dir + '/sim_mat.pickle'
      else:
        out_file = sim_dir + '/sim_mat-{}.pickle'.format(m)
      with open(out_file, 'wb') as f:
          pickle.dump(sim_mat, f)
      end = time.time()
      print('# Done in %.0f sec' % (end - start))

