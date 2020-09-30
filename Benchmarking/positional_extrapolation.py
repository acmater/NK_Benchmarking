import numpy as np
import pandas as pd
import pickle as pkl

import matplotlib.pyplot as plt

from utils.landscape_class import Protein_Landscape
from utils.sklearn_utils import collapse_concat, sklearn_split, reset_params_skorch

from copy import deepcopy


def positional_extrapolation(modeldict,
                  landscape_dict,
                  split=0.8,
                  cross_validation=1,
                  save=True,
                  file_name=None,
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

    directory : str, default="Results/"

        Directory is the directory to which the results will be saved
    """

    complete_results = {x: {key :0 for key in landscape_dict.keys()} for x in modeldict.keys()}

    for model_type, model_properties in modeldict.items():

        model, kwargs = model_properties

        for name in landscape_dict.keys():

            landscape = landscape_dict[name]

            results = []

            for i,instance in enumerate(landscape):

                positions = instance.mutated_positions

                instance_results = np.zeros((len(positions),len(positions),cross_validation))

                for fold in range(cross_validation):

                    train_datasets = []
                    test_datasets = []

                    for i,pos in enumerate(positions):

                        x_train, y_train, x_test, y_test = instance.sklearn_data(positions=positions[:i+1])
                        train_datasets.append([x_train, y_train])
                        test_datasets.append([x_test, y_test])

                    for j, p in enumerate(positions):
                        j+=1

                        x_training = collapse_concat([x[0] for x in train_datasets[:j]])
                        y_training = collapse_concat([x[1] for x in train_datasets[:j]])

                        training_losses = []
                        test_losses     = []

                        this_model = model(**kwargs)
                        if this_model.__class__.__name__ == "NeuralNetRegressor":
                            this_model.fit(x_training,y_training.reshape(-1,1))
                            print("{0} trained on Dataset {1} positions {2}".format(model_type,name,positions[:j]))
                            print()
                            for k,test_dataset in enumerate(test_datasets):
                                score = this_model.score(test_dataset[0],test_dataset[1].reshape(-1,1))
                                print("On dataset {0}, fold {1}, for positions {2}, {3} returned an R-squared of {4}".format(name, fold, positions[:k+1], model_type, score))

                                instance_results[j-1][k][fold] = score
                            reset_params_skorch(this_model) # Resets the models parameters


                        else:
                            this_model.fit(x_training,y_training)
                            print("{0} trained on Dataset {1} positions {2}".format(model_type,name,positions[:j]))
                            print()
                            for k,test_dataset in enumerate(test_datasets):
                                score = this_model.score(test_dataset[0],test_dataset[1])
                                print("On dataset {0}, fold {1} ,for positions {2}, {3} returned an R-squared of {4}".format(name, fold, positions[:k+1], model_type, score))
                                instance_results[j-1][k][fold] = score
                        print()

                results.append(instance_results.squeeze()) # Removes fold dimension if cross_validation = 1

            complete_results[model_type][name] = np.array(results)

    if save:
        if not file_name:
            file_name = input("What name would you like to save results with?")
        file = open(directory + file_name + ".pkl", "wb")
        pkl.dump(complete_results,file)
        file.close()

    return complete_results
