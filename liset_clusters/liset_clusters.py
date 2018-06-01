import numpy as np
import pandas as pd
import scipy.cluster.vq
import jenkspy
import matplotlib
import matplotlib.pyplot as plt

# Hardcoded
small = 0.1


def cluster_vector(ds, scale='linear', categories=None, missing_data='white', method='jenks'):
    """ Regroup univariate data vector as clusters for LiSET analysis

    Parameters
    ----------
    ds : a vector-like dataset (list, 1-dimensional numpy array, or pandas Series)
        The data to be clustered
    scale : str, [linear | log], default 'linear'
        Perform the clustering either on the data or on the log10() of the data, to accommodate datasets with different
        spreads
    categories : list, default (when None) is ['green', 'yellow', 'red']
        The labels to use for each cluster, defining the ranking. Can be a list of strings or numbers.
    missing_data : str, default 'white'
        Labels to identify data gaps (esp. NaN)
    method : str, ['jenks' | 'kmeans'], default 'jenks'
        Which clustering algorithm to use

    Returns
    ------
    ranking : same format as ds
        Vector of ranking, i.e., to which category every element of the ds vector belongs

    """

    # Initialize default values
    if categories is None:
        categories = ['green', 'yellow', 'red']

    # Determine the original format, just to output the ranking in the same format
    is_array = isinstance(ds, np.ndarray)
    is_list = isinstance(ds, list)

    # Internally we work with Series object
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
            raise ValueError("Cannot process negative data with the log-scale.")

        # initialize ranking with right dimensions
        ranking = data.copy()

        # zero values treated separately
        bo_gt0 = data > 0.0
        ranking[bo_gt0] = _cluster(np.log10(data[bo_gt0]), method, categories)

        # Anything with a zero is assigned to the smallest group
        ranking[~bo_gt0] = categories[0]

    else:
        raise ValueError("Expected 'linear' or 'log'." " Got '{}' instead.".format(scale))

    # Save ranking, and mark data gaps
    out[~bo_missing] = ranking
    out[bo_missing] = missing_data

    # Check for success of clustering, that all categories are present in final output
    missing_category = set(categories) - set(out)
    if missing_category:
        print("WARNING. The current clustering does not contain {}."
              " Maybe try a different clustering method".format(missing_category))

    # If initially numpy array, return as array
    if is_array:
        out = out.values
    if is_list:
        out = list(out)

    return out


def cluster_vector_plot(ds, labels=None, scale='linear', categories=None, method='jenks'):
    """ Produce bar-plot illustrating LiSET univariate data clustering of a vector of data

    A wrapper function around cluster_vector() with plotting capability

    Parameters
    ----------
    ds : a vector-like dataset (1-dimensional numpy array, or pandas Series)
        The data to be clustered
    labels: list or None
        Labels to use to identify the clustered items in the graph. If None,
        tries to extract labels from Series, or else generates a list of
        integers to serve as fallback labels.
    scale : str, [linear | log], default 'linear'
        Perform the clustering either on the data or on the log10() of the data, to accommodate datasets with different
        spreads
    categories : list of color codes
        The color to use for each cluster, defining the ranking.
        Default (when None) is ['green', 'yellow', 'red']
    method : str, ['jenks' | 'kmeans'], default 'jenks'
        Which clustering algorithm to use

    Returns
    ------
    ranking : Figure
        A bar graph colored following the ranking

    """

    # Default categories
    if categories is None:
        categories = ['green', 'yellow', 'red']

    # Perform ranking
    ranking = cluster_vector(ds, scale, categories, method)

    # If not predefined, try to extract or generate labels
    if labels is None:
        try:
            labels = ds.index
        except:
            labels = [i for i in range(len(ds))]

    # Produce graph
    fig, ax = plt.subplots()
    ax.bar(labels, ds, color=ranking)
    ax.set_xlabel('Candidates')
    ax.set_ylabel('Values')
    return fig


def cluster_matrix(df, scale='linear', categories=None, missing_data='white', method='jenks'):
    """ Regroup univariate data rows in a matrix as clusters for LiSET analysis

    Applies cluster_vector() to each ROW of a matrix

    Parameters
    ----------
    df : a matrix-like dataset (2-dimensional numpy array, or pandas DataFrame)
        The data to be clustered
    scale: str ['linear' | 'log'], or list of these trings
        Perform the clustering either on the data or on the log10() of the data, to accommodate datasets with different
        spreads. Can be specified for the whole matrix with a single string, or as a list for a per-row specification.
    categories : list, default (when None) is ['green', 'yellow', 'red']
        The labels to use for each cluster, defining the ranking. Can be a list of strings or numbers.
    missing_data : str, default 'white'
        Labels to identify data gaps (esp. NaN)
    method : str, ['jenks' | 'kmeans'], default 'jenks'
        Which clustering algorithm to use

    Returns
    ------
    ranking : array or DataFrame, depending on format of df
        The ranking, i.e., to which category each element of the df matrix belongs
    """

    # Default categories to label the clustering. By default, color code and three clusters
    if categories is None:
        categories = ['green', 'yellow', 'red']

    # Determine if is numpy array, for consistent formatting of returned output
    is_array = isinstance(df, np.ndarray)

    # Enforce DataFrame format internally
    df = pd.DataFrame(df)

    # If scale specified for table as a whole, expand this parameter to each row
    if scale == 'linear' or scale == 'log':
        scale = [scale] * df.shape[0]

    # Initialize output
    out = pd.DataFrame(index=df.index, columns=df.columns)

    # Apply row by row
    for j in range(len(df.index)):
        out.iloc[j, :] = cluster_vector(df.iloc[j, :], scale[j], categories, missing_data, method)

    # Format output
    if is_array:
        out = out.values

    return out


def cluster_matrix_plot(df, xlabels=None, ylabels=None, scale='linear', categories=None, missing_data='white', method='jenks'):
    """ Produce heat-map illustrating LiSET univariate data clustering of a matrix of data

    A wrapper function around cluster_matrix() with plotting capability

    Parameters
    ----------
    df : a matrix-like dataset (2-dimensional numpy array, or pandas DataFrame)
        The data to be clustered
    xlabels : list; default (if None) try to extract from df, or list of integers
        labels to identify the properties that define the clustering
    ylabels : list; default (if None) try to extract from df, or list of integers
        labels to identify the technologies or datapoints being clustered
    scale: str ['linear' | 'log'], or list of these strings
        Perform the clustering either on the data or on the log10() of the data, to accommodate datasets with different
        spreads. Can be specified for the whole matrix with a single string, or as a list for a per-row specification.
    categories : list of colors, default (when None) is ['green', 'yellow', 'red']
        The labels to use for each cluster, defining the ranking.
    missing_data : str, default 'white'
        Labels to identify data gaps (esp. NaN)
    method : str, ['jenks' | 'kmeans'], default 'jenks'
        Which clustering algorithm to use

    Returns
    -------
    fig : matlplotlib figure
        Heatmap with same dimensions as df

    """

    # Default categories
    if categories is None:
        categories = ['green', 'yellow', 'red']

    # Extract labels if not defined
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

    # Perform clustering with integer categories
    #   -1 for missing data
    #   0 for smallest cluster
    #   1 for next smallest cluster, etc.
    tmp_cat = [i+1 for i in range(len(categories))]
    ranking = cluster_matrix(df, scale, tmp_cat, -1, method)
    print(ranking)

    # Define color map based on category color codes
    cmap = matplotlib.colors.ListedColormap([missing_data] + categories)

    # Set bounds:
    #   -(1 + small) < -1 < -small : missing data, white
    #   -small < 0 < +small : best score, green
    #   +small < 1 < 1 + small : second best score
    #   ... etc.
    bounds = [-(1 + small), -small] + [i + small for i in tmp_cat]
    print(bounds)
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    # Save to figure
    fig, ax = plt.subplots(subplot_kw={'xticks': [], 'yticks': []})
    ax.matshow(ranking, cmap=cmap, norm=norm, interpolation=None)
    ax.set_xticklabels([''] + xlabels)
    ax.set_yticklabels([''] + ylabels)

    return fig


def _cluster(data, method, categories):
    """ Handle choice of clustering method"""

    if method == 'kmeans':
        ranking = _kmeans(data, categories)
    elif method == 'jenks':
        ranking = _jenks(data.values, categories)
    else:
        raise ValueError("Expected 'jenks' or 'kmeans' as clustering methods. Got {} instead".format(method))

    return ranking


def _kmeans(data, categories):
    """ Calculate k-means clustering

    This tiny function strives to combine the advantages of
    kmeans and kmeans2 from scipy.cluster. First, kmeans starts
    by producing an optimal set of centroids, ensuring convergence.

    Then these centroids are passed as "best_guess" to kmeans2,
    which presumably will not find better centroids than that,
    but has the advantage of returning a list linking each dataset
    to its centroid.

    Parameters
    ----------
    data : vector of data (numpy array or pandas Series)
        data to be clustered
    categories: list (of strings or integers)
        the list of cluster labels

    Returns
    -------
    sorted_categories: 1-D numpy array
        vector of sorted rankings


    """

    # Number of clusters
    k = len(categories)
    data = data.astype(float)

    # Perform clustering
    best_guess, residual = scipy.cluster.vq.kmeans(data, k)
    centroids, grouping = scipy.cluster.vq.kmeans2(data, best_guess)

    # Sort & output
    order = centroids.argsort().argsort()
    categories = np.array(categories)
    sorted_categories = categories[order]
    return sorted_categories[grouping]


def _jenks(data, categories):
    """ Calculate Jenks natural break clustering

    If same number of data points as categories (trivial), fallbacks to kmeans

    Parameters
    ----------
        data : 1D numpy array
            Data to be clustered
        categories: the list of strings or integers
            the list of cluster labels

    Returns
    -------
        out : 1-D numpy array
            vector of sorted rankings
    """

    def to_color(datum, magnitude_jenks):
        """ Based on jenks breaks, determine to which group a datum belongs"""
        i = 1
        while i < len(magnitude_jenks):
            if datum <= magnitude_jenks[i]:
                break
            i += 1
        return categories[i-1]

    try:
        # Perform clustering and get the "breaks"
        magnitude_jenks = jenkspy.jenks_breaks(data, nb_class=len(categories))

        # Ascribe each data point to its group based on these breaks
        out = np.array([to_color(i, magnitude_jenks) for i in data])

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

