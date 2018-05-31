import numpy as np
import pandas as pd
import scipy.cluster.vq
import jenkspy
import matplotlib
import matplotlib.pyplot as plt
small = 0.1


def cluster_vector(ds, scale='linear', categories=['green', 'yellow', 'red'], missing_data='white', method='jenks'):
    """ Regroup univariate data in clusters for LiSET analysis

    Args
    ----
    ds: a vector-like dataset (1-dimensional numpy array, or pandas Series)
    scale: perform the clusering either on a 'linear' or a 'log' scale
    categories: labels of the different rankings
    missing_data: labels to identify data gaps (esp. NaN)
    method: clustering algorithm, either 'jenks' or 'kmeans'

    Return
    ------
    ranking

    """

    # Initialize
    is_array = isinstance(ds, np.ndarray)
    is_list = isinstance(ds, list)
    ds = pd.Series(ds)
    out = pd.Series(index=ds.index)

    # Filter out NaN
    bo_missing = pd.isnull(ds)
    data = ds[~bo_missing]

    # regroup in clusters
    if scale == 'linear':
        ranking = _cluster(data, method, categories)
    elif scale == 'log':

        # Negatives not allowed
        if np.any(data < 0.0):
            raise ValueError("Cannot process negative data")

        # initialize ranking with right dimensions
        ranking = data.copy()

        # zero values treated separately
        bo_gt0 = data > 0.0
        ranking[bo_gt0] = _cluster(np.log10(data[bo_gt0]), method, categories)

        # Anything with a zero is assigned to the smallest group
        ranking[~bo_gt0] = categories[0]

    else:
        raise ValueError("Expected 'linear' or 'log'." " Got '{}' instead.".format(scale))

    # Identify lack of data
    out[~bo_missing] = ranking
    out[bo_missing] = missing_data

    # Check for success of clustering
    missing_category = set(categories) - set(out)
    if missing_category:
        print("WARNING. The current clustering does not contain {}. Maybe try a different clustering method".format(missing_category))

    # If initially numpy array, return as array
    if is_array:
        out = out.values
    if is_list:
        out = list(out)

    return out

def cluster_vector_plot(ds, labels=None, scale='linear', categories=['green', 'yellow', 'red'], method='jenks'):
    ranking = cluster_vector(ds, scale, categories, method)
    if labels is None:
        try:
            labels = ds.index
        except:
            labels = [i for i in range(len(ds))]
    fig, ax = plt.subplots()
    ax.bar(labels, ds, color=ranking)
    ax.set_xlabel('Candidates')
    ax.set_ylabel('Values')
    return fig



def cluster_matrix(df, scale='linear', categories=['green', 'yellow', 'red'],
        missing_data='white', method='jenks'):
    """ Applies cluster_vector() to each row of a matrix

    Args
    ----

    df: a matrix-like dataset (2-dimensional numpy array, or pandas DataFrame)
    scale: perform the clusering either on a 'linear' or a 'log' scale
            Can be specified for whole matrix, or as a list for a per-row
            specification.
    categories: labels of the different rankings
    missing_data: labels to identify data gaps (esp. NaN)
    method: clustering algorithm, either 'jenks' or 'kmeans'

    """

    # Enforce DataFrame format internally
    is_array = isinstance(df, np.ndarray)
    df = pd.DataFrame(df)

    # If scale specified for each table, expand this parameter to each row
    if scale == 'linear' or scale == 'log':
        scale = [scale] * df.shape[0]

    # Initialize output
    out = pd.DataFrame(index=df.index, columns=df.columns)

    # Apply column by column
    for j in range(len(df.index)):
        out.iloc[j, :] = cluster_vector(df.iloc[j, :], scale[j], categories,
                missing_data, method)
    if is_array:
        out = out.values

    return out

def cluster_matrix_plot(df, xlabels=None, ylabels=None, scale='linear', categories=['green', 'yellow', 'red'],
        missing_data='white', method='jenks'):

    if xlabels is None:
        try:
            xlabels = list(df.columns)
        except:
            xlabels = [i for i in range(df.shape[1])]


    if ylabels is None:
        try:
            ylabels = list(df.index)
        except:
            ylabels = [i for i in range(df.shape[0])]

    tmp_cat = [i+1 for i in range(len(categories))]
    ranking = cluster_matrix(df, scale, tmp_cat, -1, method)
    print(ranking)

    cmap = matplotlib.colors.ListedColormap(categories)
    bounds = [0,] + [i + 0.1 for i in tmp_cat]
    print(bounds)
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)



    fig, ax = plt.subplots(subplot_kw={'xticks': [], 'yticks': []})
    ax.matshow(ranking, cmap=cmap, norm=norm, interpolation=None)
    ax.set_xticklabels(['']+ list(df.columns))
    ax.set_yticklabels(['']+ list(df.index))

    return fig



def _kmeans(data, categories):
    """ Calculate k-means clustering
    Args
    -----
        data :  data to be clustered
        categories: the list of cluster labels (can be strings or integers)

    This tiny function strives to combine the advantages of 
    kmeans and kmeans2 from scipy.cluster. First, kmeans starts
    by producing an optimal set of centroids, ensuring convergence.

    Then these centroids are passed as "best_guess" to kmeans2, 
    which presumably will not find better centroids than that, 
    but has the advange of returning a list linking each dataset 
    to its centroid.
    """

    k = len(categories)
    data = data.astype(float)
    best_guess, residual = scipy.cluster.vq.kmeans(data, k)
    centroids, grouping = scipy.cluster.vq.kmeans2(data, best_guess)

    order = centroids.argsort().argsort()
    categories = np.array(categories)
    sorted_categories = categories[order]
    return sorted_categories[grouping]


def _jenks(data, categories):
    """ Calculate Jenks natural break clustering

    Args
    ----
        data :  data to be clustered
        categories: the list of cluster labels (can be strings or integers)
    """

    def to_color(datum, magnitude_jenks):
        i = 1
        found = False
        while i < len(magnitude_jenks):
            if datum <= magnitude_jenks[i]:
                break
            i += 1
        return(categories[i-1])


    try:
        magnitude_jenks=jenkspy.jenks_breaks(data,nb_class=len(categories))
        out =  np.array([to_color(i, magnitude_jenks) for i in data ])
    except ValueError as err:
        # If data sample has exactly as many data points as the number of
        # categories, then obviously each datapoint gets its own category.
        # However, such a trivial case is too simple for the Jenks algorithm,
        # so falling back to kmeans to find the obvious answer.
        if len(data) == len(categories):
            out = _kmeans(data, categories)
        else:
            raise err

    return out

def _cluster(data, method, categories):
    """ Handle choice of clustering method"""

    if method == 'kmeans':
        ranking = _kmeans(data, categories)
    elif method == 'jenks':
        ranking = _jenks(data.values, categories)
    else:
        raise ValueError("Expected 'jenks' or 'kmeans' as clustering methods."
                " Got {} instead".format(method))

    return ranking



