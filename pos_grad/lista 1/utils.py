import matplotlib.pyplot as plt
import numpy as np
import itertools
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from typing import List, Any
from torch.utils.data import Dataset, DataLoader 



class CustomDataset(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)





def plot_confusion_matrix(cm: Any,
                          target_names: List[str] = None,
                          title: str ='Confusion matrix',
                          cmap: Any = None,
                          normalize: bool = False):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    
    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if target_names is None:
        target_names = [i for i in range(cm.shape[0])]

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()

def piecewise_norm(vec,val,n):
    assert val > min(vec)
    
    return np.where(vec < val, n*(vec - min(vec))/(val - min(vec)), (1-n)*(vec - val)/(max(vec) - val) + n)

def transform_data(dataset: pd.DataFrame, normalize: bool = True):
    """
    Transforms data from a pandas DataFrame into a new DataFrame.
    Parameters:
        dataset (pd.DataFrame): The input dataset
        normalize (bool): Boolean indicating whether to normalize numerical data. Default: True
    Returns:
        pd.DataFrame: The transformed dataframe
    """

    new_df = pd.DataFrame()

    #Varre todas as colunas do dataframe.
    for name, name_type in dataset.dtypes.items():

        if name == 'target':
            new_df[name] = dataset[name].values

        #Tratamento de dados utilizando o atributo categórico. No caso, é utilizada a codificação 1 of N.
        elif name_type == object:
            raw_data = dataset[name].values
            d_encoder = LabelEncoder()
            d_encoder.fit(raw_data)
            d_encoded = d_encoder.transform(raw_data)
            #dummy_y = to_categorical(d_encoded)
            dim = len(d_encoder.classes_)
            dummy_y = np.eye(dim)[d_encoded]
            
            for (j,k) in enumerate(d_encoder.classes_):
                new_df[f'{name}_{k}'] = dummy_y[:,j].astype('int')
        
        
        #Caso numérico, utilizando normalização min-max
        else:
            raw_data = dataset[name].values
            if normalize:
                new_df[name] = (raw_data - min(raw_data))/(max(raw_data) - min(raw_data))
            else:
                new_df[name] = raw_data
            
                        
    return new_df