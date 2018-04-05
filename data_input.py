from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import sys

import numpy as np

import inception_input

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes



class DataSet(object):

  def __init__(self,
               images,
               labels,
               ids,
               training_sim=None,
               ref_order_ids=None,
               label_matrix=None,
               dtype=dtypes.float32,
               reshape=True):

    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    assert images.shape[0] == labels.shape[0], (
      'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
    assert images.shape[0] == ids.shape[0], (
        'images.shape: %s ids.shape: %s' % (images.shape, ids.shape))
    self._num_examples = images.shape[0]

    # Convert shape from [num examples, rows, columns, depth]
    # to [num examples, rows*columns] (assuming depth == 1)
    if reshape:
      assert images.shape[3] == 1
      images = images.reshape(images.shape[0],
                              images.shape[1] * images.shape[2])
    if dtype == dtypes.float32:
      # Convert from [0, 255] -> [0.0, 1.0].
      images = images.astype(np.float32)
      images = np.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._ids = ids
    self._training_sim = training_sim
    self._ref_order_ids = ref_order_ids
    self._label_matrix = label_matrix
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def ids(self):
      return self._ids

  @property
  def training_sim(self):
      return self._training_sim

  @property
  def ref_order_ids(self):
      return self._ref_order_ids

  @property
  def label_matrix(self):
      return self._label_matrix

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""

    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      self._ids = self._ids[perm]
      try: # Training
        self._training_sim = self._training_sim[perm]
      except TypeError:  # Validation
        pass
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch

    try: # Training
        sim_batch = self._training_sim[start:end]
    except TypeError: # Validation
        sim_batch = None

    return self._images[start:end], sim_batch, self._labels[start:end], self._ids[start:end]




def load_images(dir, max_n_images=sys.maxsize):

    transfer_values, ids = inception_input.cache_images_dir(dir, max_n_images)

    return {'images': transfer_values, 'id': ids} # Images and ids in the same order




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


def create_one_hot_label_vector(data, dir, lookup_table):
    link = dir + '/pickle/combined.pickle'
    with open(link, 'rb') as f:
        all_txt = pickle.load(f)
    res = []
    for id in data['id']:
        res_i = np.zeros(len(lookup_table), dtype=np.float32)
        labels_item = all_txt[id] # List of (label, confidence)
        for l in labels_item: # Tuple (label, confidence)
            try:
                res_i[lookup_table[l[0]]] = l[1] # One-hot vector
            except KeyError:
                print('%s is not in the training set, it has been discarded.' % l[0])
        res.append(res_i)
    res = np.array(res)
    return res # Order matches order of ids and images in data


def get_label_dict(train_dir, val_dir):
    """ Return the dictionnary ID - (labels, confidences) """
    link = train_dir + '/pickle/combined.pickle'
    with open(link, 'rb') as f:
        dict_train = pickle.load(f)
    if val_dir != None:
        link = val_dir + '/pickle/combined.pickle'
        with open(link, 'rb') as f:
            dict_val = pickle.load(f)
        return dict(dict_train, **dict_val)
    else:
        return dict_train


def score(label_dict, target='', selection=list(), n=50):
    """
    Calculate the score of a selected set compared to the target image.
    :param label_dict: dictionary of labels, keys are image IDs
    :param target: image ID of the query image
    :param selection: the list of IDs retrieved
    :param n: the assumed number of relevant images. Kept fixed at 50
    :return: the calculated score
    """
    # Remove the queried element
    selection = list(set(selection) - {target})

    # k is the number of retrieved elements
    k = len(selection)
    if target in label_dict.keys():
        target_dict = dict(label_dict[target])
    else:
        print("Couldn't find " + target + " in the dict keys.")
        target_dict = {}

    # Current score will accumulate the element-wise scores,
    # before rescaling by scaling by 2/(k*n)
    current_score = 0.0

    # Calculate best possible score of image
    best_score = sum(target_dict.values())

    # Avoid problems with div zero. If best_score is 0.0 we will
    # get 0.0 anyway, then best_score makes no difference
    if best_score == 0.0:
        best_score = 1.0

    # Loop through all the selected elements
    for selected_element in selection:

        # If we have added a non-existing image we will not get
        # anything, and create a dict with no elements
        # Otherwise select the current labels
        if selected_element in label_dict.keys():
            selected_dict = dict(label_dict[selected_element])
        else:
            print("Couldn't find " + selected_element +
                  " in the dict keys.")
            selected_dict = {}

        # Extract the shared elements
        common_elements = list(set(selected_dict.keys()) &
                               set(target_dict.keys()))
        if len(common_elements) > 0:
            # for each shared element, the potential score is the
            # level of certainty in the element for each of the
            # images, multiplied together
            element_scores = [selected_dict[element] *
                              target_dict[element]
                              for element in common_elements]
            # We sum the contributions, and that's it
            current_score += sum(element_scores) / best_score
        else:
            # If there are no shared elements,
            # we won't add anything
            pass

    # We are done after scaling
    return current_score * 2 / (k + n)



def read_data_sets(train_dir,
                   val_dir,
                   sim_dir,
                   max_n_images,
                   dtype=dtypes.uint8,
                   reshape=False):

  # Load reference table
  with open(sim_dir + '/sim_mat.pickle', 'rb') as f:
      sim_mat = pickle.load(f)

  # Load list ids
  with open(sim_dir + '/sim_ids.pickle', 'rb') as f:
      list_ids = pickle.load(f)

  # Load lookup table
  with open(sim_dir + '/sim_lookup_table.pickle', 'rb') as f:
      lookup_table = pickle.load(f)

  # Load training images
  training_data = load_images(train_dir, max_n_images)

  # Load validation images
  if val_dir != None:
    val_data = load_images(val_dir, max_n_images)

  # Get training labels (needed for F-score)
  train_labels = create_one_hot_label_vector(training_data, train_dir, lookup_table)

  # Get validation labels (needed for F-score)
  if val_dir != None:
    val_labels = create_one_hot_label_vector(val_data, val_dir, lookup_table)

  # Training table (needed for F-score)
  labels_tr_mat = train_labels

  # Reference order number - ID
  ref_order_ids = list_ids

  # Similarity vector ### list_ids and sim_mat should consider same order of training images !
  training_sim = []
  for i in range(max_n_images):
      training_sim.append(sim_mat[i,:])
  training_sim = np.array(training_sim)

  # Create datasets
  train = DataSet(training_data['images'],
                  train_labels,
                  training_data['id'],
                  training_sim=training_sim,
                  ref_order_ids=ref_order_ids,
                  label_matrix=labels_tr_mat,
                  dtype=dtype,
                  reshape=reshape)

  if val_dir != None:
    validation = DataSet(val_data['images'],
                       val_labels,
                       val_data['id'],
                       dtype=dtype,
                       reshape=reshape)
    return base.Datasets(train=train, validation=validation, test=validation)
  else:
    return base.Datasets(train=train, validation=train, test=train)

