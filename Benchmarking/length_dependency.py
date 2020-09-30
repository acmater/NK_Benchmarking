import numpy as np
import pickle as pkl

from utils.landscape_class import Protein_Landscape
from utils.sklearn_utils import train_test_model

def length_testing(modeldict,
                   landscape_dict,
                   split=0.8,
                   save=True,
                   AAs="ACDEFGHIKL",
                   seq_lens=[10, 50, 100, 200, 300, 400, 500],
                   file_name=None,
                   directory="Results/"):
    """
    Length testing function that takes a dictionary of models, a landscape dictionary,
    and a list of sequence lengths. It iterates over all of these and leverages the
    Protein Landscape function that enables it to randomly inject length into its
    sequences to train each model on each of these values.

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

    AAs : str, default="ACDEFGHIKL"

        List of possible amino acids that is passed to the length extension function

    seq_lens : list, default=[10,50,100,200,300,400,500]

        List of sequence lengths, determining how long the extended sequences will be.

    file_name : str, default=None

        File name to use if saving file. If none is provided, user will be prompted for one.

    directory : str, default="Results/"

        Directory is the directory to which the results will be saved
    """

    complete_results = {x: {key :0 for key in landscape_dict.keys()} for x in modeldict.keys()}

    for model_type, model_properties in modeldict.items():
        # Iterate over model types
        model, kwargs = model_properties

        for name in landscape_dict.keys():
            # Iterate over each landscape
            results = np.zeros((len(landscape_dict[name]),len(seq_lens)))

            for i,instance in enumerate(landscape_dict[name]):
                # Iterate over each INSTANCE of each landscape, 1 for experimental
                for j,length in enumerate(seq_lens):

                    temp_model = model(**kwargs)

                    x_train, y_train, x_test, y_test = instance.return_lengthened_data(50,AAs=AAs,split=split)

                    score = train_test_model(temp_model,x_train,y_train,x_test,y_test)

                    results[i][j] = score

                    print("For sequence length {0}, {1} returned an R-squared of {2}".format(length, model_type, score))

            complete_results[model_type][name] = results

    if save:
        if not file_name:
            file_name = input("What name would you like to save results with?")
        file = open(directory + file_name + ".pkl", "wb")
        pkl.dump(complete_results,file)
        file.close()

    return complete_results
