import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.rcParams['text.usetex'] = True

def cv_plot(xticks, train, test, ylabel, xlabel, title):
    plt.errorbar(xticks, train[0], yerr=train[1], ecolor='b', capsize=3,
                 label="train")
    plt.errorbar(xticks, test[0], yerr=test[1], ecolor='r', capsize=3,
                label='test')

    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xticks(xticks)
    plt.title(title)
    plt.legend()
    plt.show()



def autolabel(ax, rects, xpos='center'):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0, 'right': 1, 'left': -1}

    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(offset[xpos]*3, 3),  # use 3 points offset
                    textcoords="offset points",  # in both directions
                    ha=ha[xpos], va='bottom')
    
def plot_smooths():
    raw_means, raw_std = (0.6691, 0.7115), (0.0192, 0.0153)
    pos_means, pos_std = (0.4219, 0.4219), (0.0337, 0.0337)

    width = 0.35
    ind = np.arange(len(raw_means))
    fig, ax = plt.subplots()
    rect1 = ax.bar(ind - width/2, pos_means, width, yerr=pos_std, capsize=3,
                   label='POS')

    rect2 = ax.bar(ind + width/2, raw_means, width, yerr=raw_std, capsize=3,
                   label='raw')

    ax.set_ylabel('$f_1$ score')
    ax.set_title('Comparision of the two smoothing techniques for raw and POS models')
    ax.set_xticks(ind)
    ax.set_xticklabels(['Technique 1', 'Technique 2'])
    ax.legend()

    autolabel(ax, rect1, 'left')
    autolabel(ax, rect2, 'right')

    fig.tight_layout()
    plt.show()



def plot_baseline_model_comparison():
    raw_means, raw_std = (0.7115,), (0.0153,)
    base_means, base_std = (0.4651,), (0.2731,)


    width = 0.35
    ind = np.arange(len(raw_means))
    fig, ax = plt.subplots()
    rect1 = ax.bar(ind - width/2, raw_means, width, yerr=raw_std, capsize=3,
                   label='raw')

    rect2 = ax.bar(ind + width/2, base_means, width, yerr=base_std, capsize=3,
                   label='baseline')

    ax.set_ylabel('$f_1$ score')
    ax.set_title('Comparision of our model with the baseline')
    ax.set_xticks(ind)
    ax.set_xticklabels([])
    ax.legend()

    autolabel(ax, rect1, 'left')
    autolabel(ax, rect2, 'right')

    fig.tight_layout()
    plt.show()    
