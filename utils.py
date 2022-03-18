from collections import OrderedDict

import numpy as np


def vector2dict(input_vector, initial_tensor_shapes_dict):
    slice_start_idx = 0
    result_dict = dict()
    for key in initial_tensor_shapes_dict:
        dim_1, dim_2 = initial_tensor_shapes_dict[key][0], initial_tensor_shapes_dict[key][1]
        slice_end_idx = slice_start_idx + (dim_1 * dim_2)
        vector_slice = input_vector[slice_start_idx: slice_end_idx]
        matrix = vector_slice.reshape(dim_1, dim_2)
        result_dict[key] = matrix
        slice_start_idx = slice_end_idx
    return result_dict


def dict2vector(input_dict):
    key2shape_dict = OrderedDict()
    vector_list = []
    for key in input_dict:
        vector, shape = input_dict[key].flatten(), input_dict[key].shape
        key2shape_dict[key] = shape
        vector_list.append(vector)
    result_vector = np.concatenate(vector_list, axis=0)
    return result_vector, key2shape_dict


def test():
    initial_dict = {
        'w_1': np.random.rand(2, 3),
        'b_1': np.random.rand(1, 3),
        'w_2': np.random.rand(3, 2),
        'b_2': np.random.rand(1, 2)
    }
    print(initial_dict)
    dict_vector, shape_dict = dict2vector(initial_dict)
    print(dict_vector)
    result_dict = vector2dict(dict_vector, shape_dict)
    for key in initial_dict:
        if (initial_dict[key] == result_dict[key]).all() == False:
            print('Dicts not equal!')
            return
    print('Dicts equal!')


test()
