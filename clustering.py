from sklearn.metrics import precision_score, accuracy_score
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import seaborn as sns
import numpy as np
from munkres import Munkres

sns.set_theme()


def make_cost_matrix(predicted_labels, true_labels):
    """
    """
    uc1 = np.unique(predicted_labels)
    uc2 = np.unique(true_labels)
    l1 = uc1.size
    l2 = uc2.size
    assert (l1 == l2 and np.all(uc1 == uc2))

    m = np.ones([l1, l2])
    for i in range(l1):
        it_i = np.nonzero(predicted_labels == uc1[i])[0]
        for j in range(l2):
            it_j = np.nonzero(true_labels == uc2[j])[0]
            m_ij = np.intersect1d(it_j, it_i)
            m[i, j] = -m_ij.size
    return m


def reassign_labels(predicted_labels, true_labels):
    """Reassign labels to achieve best mapping

    Args:
        predicted_labels (list[int]): predicted labels
        true_labels (list[int]): true labels

    Returns:
        new_labels (list[int]): best mapping of predicted labels to true labels
    """
    cost_matrix = make_cost_matrix(predicted_labels=predicted_labels, true_labels=true_labels)
    m = Munkres()
    indexes = m.compute(cost_matrix)
    mapper = {old: new for (old, new) in indexes}
    new_labels = translate_clustering(predicted_labels, mapper)
    return new_labels


def translate_clustering(predicted_labels, mapper):
    """Map labels

    Args:
        predicted_labels (list[int]): predicted labels
        mapper (dict): mapping of old indexes to new ones

    Returns:
        new_labels (array): new labels
    """
    new_labels = np.array([mapper[i] for i in predicted_labels])
    return new_labels


def perform_clustering(embeddings):
    """Perform k-means clustering on embeddings

    Notes:
        n_clusters=2 since we want to differentiate between product urls and non-product urls

    Args:
        embeddings (array): url embeddings

    Returns:
        kmeans_labels (list[int]): labels predicted by k-means algorithm
    """
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans_labels = kmeans.fit_predict(embeddings)
    return kmeans_labels


def perform_dimensionality_reduction(embeddings):
    """Perform t-SNE dimensionality reduction

    Args:
        embeddings (array): url embeddings

    Returns:
        data_tsne (array): embeddings transformed to 2D space (from 100D)
    """
    tsne = TSNE(n_components=2, random_state=42, perplexity=10)
    data_tsne = tsne.fit_transform(embeddings)
    return data_tsne


def plot_data(data, true_labels, predicted_labels):
    """Plots data results

    Args:
        data (array): embeddings in 2D space
        true_labels (list[int]): true labels of urls
        predicted_labels (array): predicted labels of urls

    Returns:

    """
    accuracy, precision = calculate_accuracy_precision(true_labels, predicted_labels)
    plt.figure(figsize=(10, 7))
    # Plot the data with cluster labels
    plt.scatter(data[:, 0], data[:, 1], c=predicted_labels, cmap='Spectral')
    plt.suptitle('t-SNE Visualization with Clusters (K-means)')
    plt.title(f'Accuracy:{accuracy} - Precision: {precision}')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.show()


def calculate_accuracy_precision(true_labels, predicted_labels):
    """Calculate the accuracy and precision

    Args:
        true_labels (list[int]): true labels of urls
        predicted_labels (array): predicted labels of urls

    Returns:
        accuracy (float): accuracy of model
        precision (float): precision of model
    """
    accuracy = round(accuracy_score(true_labels, predicted_labels), 3)
    precision = round(precision_score(true_labels, predicted_labels), 3)
    return accuracy, precision
