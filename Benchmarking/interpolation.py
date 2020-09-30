import numpy as np
import pickle as pkl

from utils.landscape_class import Protein_Landscape
from utils.sklearn_utils import train_test_model

def interpolation(modeldict,
                  landscape_dict,
                  split=0.8,
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

            results = np.zeros((len(landscape_dict[name]),))

            for i,instance in enumerate(landscape_dict[name]):

                temp_model = model(**kwargs)
                x_train, y_train, x_test, y_test = instance.sklearn_data(split=split)

                score = train_test_model(temp_model,x_train,y_train,x_test,y_test)
                print("{0} trained on Dataset {1} achieved an R^2 of {2}".format(model_type,name,score))
                results[i] = score

            complete_results[model_type][name] = results

    if save:
        if not file_name:
            file_name = input("What name would you like to save results with?")
        file = open(directory + file_name + ".pkl", "wb")
        pkl.dump(complete_results,file)
        file.close()

    return complete_results
