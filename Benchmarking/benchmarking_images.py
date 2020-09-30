import matplotlib.pyplot as plt
import numpy as np

def highlight_cell(x,y, ax=None, **kwargs):
    rect = plt.Rectangle((x-.5, y-.5), 1,1, fill=False, **kwargs)
    ax = ax or plt.gca()
    ax.add_patch(rect)
    return rect

def highlight_extrapolation(seq_len):
    """
    Generates highlights for each box which represents extrapolation
    """
    for x,y in zip(range(1,seq_len),range(0,seq_len-1)):
        highlight_cell(x,y,color="black", linewidth=3)

def generate_extrapolation_plot(complete_results,model):
    """
    Generate a grid of correlation heatmaps with extrapolation highlighted for each ruggedness value given.
    This code was written for four ruggedness values, it is up to you to ensure that the plot makes sense
    if an odd number of ruggedness values are given.

    model [str] : Describes what type of model architecture was used to produce results. Used in title
                  and save file name.

    complete_results [np.array | shape(n,r,r)]
        where:
            n is the number of ruggedness values - 1
            r is the length of the sequence
    """
    row = 1
    col = complete_results.shape[-1]

    complete_results = complete_results.clip(min=0) # Remove negative values to keep heatmap clean

    if len(complete_results.shape) > 3:
        complete_results = np.mean(complete_results,axis=0) # Averages all instances if not already done so.

    seq_len = complete_results.shape[-1]
    plt.subplots_adjust(left=0.2, bottom=None, right=3, top=None, wspace=None, hspace=None)
    fig = plt.figure(figsize=(col*3,4))

    axes = []
    for val in range(complete_results.shape[0]):
        axes.append(fig.add_subplot("{0}{1}{2}".format(row,col,val+1)))
        axes[val].imshow(complete_results[val],cmap="Reds",vmin=0,vmax=1) # Squared to produce R2
        highlight_extrapolation(seq_len)

    for i,ax in enumerate(axes):

        ax.set_title("K = {}".format(i),fontsize=10)
        ax.set_xlabel("Hamming Distance \n from Seed Sequence",fontsize=10)

        xticklabs = [-1,1,3,5,7]#["{}".format(x) for x in range(0,seq_len+1)] NEED TO FIX THIS LINE AS IT IS CHEATING TO MAKE THE PLOT
        ax.set_xticks = np.arange(seq_len)
        ax.set_xticklabels(xticklabs)

        if i == 0:
            ax.set_ylabel("Hamming Distance training data used",fontsize=10)

            lengths = [x+1 for x in range(seq_len+1)]
            yticklabs = [",".join(map(str,lengths[:i])) for i in range(len(lengths))]
            ax.set_yticks = np.arange(seq_len)
            ax.set_yticklabels(yticklabs)
        else:
            ax.get_yaxis().set_visible(False)

    plt.suptitle("Model Performance on NK Landscapes with Varying Ruggedness (K) - {} Model".format(model),fontsize=16)
    plt.savefig("Images/Model_Performance_NK_Landscapes_{}.png".format(model),dpi=400,bbox_inches="tight")
    plt.show()

def gen_interpolation_plot(complete_results,name="Interpolation.png"):
    """
    Complete_results [dict]  -- A dictionary with keys as names of model
    architectures as values an (insance x ruggedness) array of correlations
    """

    labels = []
    arrays = []

    for key,val in complete_results.items():
        labels.append(key)
        arrays.append(val)
        max_rug = complete_results[key].shape[-1]

    arrays = np.mean(np.array(arrays),axis=1)
    arrays = arrays.clip(min=0) # Remove negative values to keep heatmap clean

    fig, ax = plt.subplots()
    im = ax.imshow(arrays,vmin=0,vmax=1,cmap="Reds")

    ax.set_xticks(np.arange(max_rug))
    ax.set_xticklabels([x for x in range(max_rug)],fontsize=16)

    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels,fontsize=16)

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("$R^2$ Correlation", rotation=-90, va="bottom",fontsize=16)

    ax.set_title("Interpolation Performance for each \n model with varying Ruggedness",fontsize=16,pad=20)

    plt.savefig(name,bbox_inches="tight",dpi=300)
    plt.show()



def plot_seq_results(results, models, max_rug=5, figsize=(7,3), rows=1, cols=2,
                 seq_lens=[10, 50, 100, 200, 300, 400, 500], fontsize=13, tickfontsize=10,
                 **kwargs):
    fig, axs = plt.subplots(rows,cols, figsize=figsize)
    plt.subplots_adjust(**kwargs)

    for i, ax in enumerate(axs):
        model, data = list(results.items())[i]
        means = np.mean(data, axis=0)
        ax.imshow(means, cmap="Reds", vmin=0, vmax=1)
        if i==0:
            ax.set_ylabel('Ruggedness (K) Value', fontsize=fontsize)

        ax.set_xlabel('Sequence length', fontsize=fontsize)
        ax.set_title(models[i], fontsize=fontsize)
        ax.set_xticklabels([0]+seq_lens, fontsize=tickfontsize)
        ax.set_yticklabels([0]+list(range(max_rug)), fontsize=tickfontsize)
    plt.savefig("Sequence_Results.png")
    plt.show()


def plot_ablation_results(results, models, max_rug=5, figsize=(7,3), rows=1, cols=2,
                 sampleDensities=[0.1, 0.25, 0.5, 0.75, 0.9, 1], fontsize=13, tickfontsize=10,
                 **kwargs):
    fig, axs = plt.subplots(rows,cols, figsize=figsize)
    plt.subplots_adjust(**kwargs)

    for i, ax in enumerate(axs):
        model, data = list(results.items())[i]
        means = np.mean(data, axis=0)
        ax.imshow(means, cmap="Reds", vmin=0, vmax=1)
        if i==0:
            ax.set_ylabel('Ruggedness (K) Value', fontsize=fontsize)

        ax.set_xlabel('Sampling density', fontsize=fontsize)
        ax.set_title(models[i], fontsize=fontsize)
        ax.set_xticklabels([0]+sampleDensities, fontsize=tickfontsize)
        ax.set_yticklabels([0]+list(range(max_rug)), fontsize=tickfontsize)
    plt.show()
