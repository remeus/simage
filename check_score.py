import numpy as np

from start import score



def check_f_score(data, label_dict):
    f_score_train = 0
    for i in range(data.train.images.shape[0]):
        sim_vec = data.train.training_sim[i]
        sim_images_index = np.where(sim_vec)[0]
        sim_images_ids = list(data.train.ref_order_ids[sim_images_index])
        f_score_i = score(label_dict, target=data.train.ids[i], selection=sim_images_ids, n=50)
        f_score_train += f_score_i
    f_score_train /= data.train.images.shape[0]
    return f_score_train
