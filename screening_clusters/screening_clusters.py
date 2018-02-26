import numpy as np
import pandas as pd
import scipy.cluster.vq

def cluster_series(ds, dist='kmeans', categories=['green', 'yellow', 'red']):

    # Initialize
    out = pd.Series(index=ds.index)

    # Filter out negatives
    bo = ds >=0

    # regroup in clusters
    imps = ds[bo]
    if dist == 'kmeans':
        ranking = kmeans(imps, categories)[1]
    elif dist == 'kmeanslog':
        ranking = kmeanslog(imps, categories)[1]
    else:
        raise ValueError("Expected 'kmeans' or 'kmeanslog'. Got '{}' instead.".format(dist))

    # Identify lack of data
    out[bo] = ranking
    out[~bo] = 'white'

    return out


def cluster_dataframe(df, dist, categories=['green', 'yellow', 'red']):

    # Initialize
    out = pd.DataFrame(index=df.index, columns=df.columns)

    # Apply column by column
    for j in range(len(df.columns)):
        out.iloc[:, j] = cluster_series(df.iloc[:,j], dist[j], categories)
    return out


def kmeans(data, categories=['green', 'yellow', 'red']):
    """ Calculate k-means clustering
    Args
    -----
        data :  data to be clustered

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
    grouping = to_colors(centroids, grouping, categories)

    return np.sort(centroids), grouping

def kmeanslog(data, categories=['green', 'yellow', 'red']):

    k = len(categories)

    # Initial definitions
    score = data.copy()

    # Negatives not allowed
    if np.any(data < 0.0):
        raise ValueError("Cannot process negative data")

    # Null values treated separately
    bo = data > 0.0
    abs_pos = data[bo]

    # Perform kmeans on log10 on absolutely positive values
    centroids, grouping = kmeans(np.log10(abs_pos), categories)
    score[bo] = grouping

    # Anything with a zero is assigned to the smallest centroid
    score[~bo] = categories[0]
    return centroids, grouping


def to_colors(centroids, groups, colors=['green', 'yellow', 'red']):
    order = centroids.argsort().argsort()
    colors = np.array(colors)
    sorted_colors = colors[order]
    return sorted_colors[groups]

#def tocolor(i, means):
#    val = means[i]
#    if val == np.max(means):
#        color = 'red'
#    elif val == np.min(means):
#        color = 'green'
#    else:
#        color = 'yellow'
#    return color


