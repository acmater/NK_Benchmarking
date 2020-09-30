import numpy as np
from utils.sklearn_utils import *

import pickle as pkl

def ablation_testing(modeldict,
                     landscape_dict,
                     split=0.8,
                     cross_validation=1,
                     save=True,
                     file_name=None,
                     shuffle=True,
                     sample_densities=[0.9, 0.7, 0.5, 0.3, 0.1],
                     directory="Results/"):
    """
    Interpolation function that takes a dictionary of models and a landscape dictionary
    and iterates over all models and landscapes, recording results, before finally (saving)
    and returning them.

    Parameters
    ----------
    modeldict : dict

        Dictionary of model architectures. Format: {sklearn.model : **kwargs}

    landscape_dict : dict

        Dictionary of protein landscapes. Format : {Name : [Protein_Landscape()]}

    split : float, default=0.8, Allowed values: 0 < split < 1

        The split point used to partition the data

    cross_validation : int, default=1

        The number of times to randomly resample the dataset, typically used with
        experimental datasets.

    save : Bool, default=True

        Boolean value used to determine whether or not the file will be saved

    file_name : str, default=None

        File name to use if saving file. If none is provided, user will be prompted for one.

    sample_densities : list, default=[0.9, 0.7, 0.5, 0.3, 0.1]

        Split densities that are passed to the sklearn_data function of each landscape

    directory : str, default="Results/"

        Directory is the directory to which the results will be saved
    """

    complete_results = {x: {key :0 for key in landscape_dict.keys()} for x in modeldict.keys()}

    for model_type, model_properties in modeldict.items():
        # Iterate over model types
        model, kwargs = model_properties

        for name in landscape_dict.keys():
            # Iterate over each landscape
            results = np.zeros((len(landscape_dict[name]),len(sample_densities),cross_validation))

            for i,instance in enumerate(landscape_dict[name]):
                # Iterate over each INSTANCE of each landscape, 1 for experimental

                for fold in range(cross_validation):
                    print()
                    for j,density in enumerate(sample_densities):

                        temp_model = model(**kwargs)

                        x_train, y_train, x_test, y_test = instance.sklearn_data(split=0.8,shuffle=shuffle)
                        idxs = np.random.choice(len(x_train),size=int(len(x_train)*density))
                        actual_x_train = x_train[idxs]
                        actual_y_train = y_train[idxs]

                        score = train_test_model(temp_model,actual_x_train,actual_y_train,x_test,y_test)

                        results[i][j][fold] = score

                        print("For sample density {0}, on {1} instance {2} {3} returned an R-squared of {4}".format(density, name, i, model_type, score))

            complete_results[model_type][name] = results.squeeze()

    if save:
        if not file_name:
            file_name = input("What name would you like to save results with?")
        file = open(directory + file_name + ".pkl", "wb")
        pkl.dump(complete_results,file)
        file.close()

    return complete_results
