import numpy as np

def bootstrap_negatives(dfs, neg, neg_old):
    """
    Resample from a set of negative labels, emphasizng
    examples that are currently mis-classified

    Parameters
    ----------
    dfs : dict
        Decision functions for 'neg', 'cv_pos', and 'neg_old'
    neg : list
        A set of negative examples
    neg_old : list
        The negative examples used to train the classifier

    Returns
    -------
    A resampled list of negative values

    Notes
    -----
    This function creates a new version of `neg` that retains
    the half of the old examples with the highest decision function.
    The other half are drawn randomly from the portion of `neg`
    which have decision values >= the 10th percentile of the
    decision function of `cv_pos`.

    This results in a new set of training data that emphasizes
    examples which are currently poorly-classified

    """

    df = dfs['neg']
    df_pos = dfs['cv_pos']
    df_pos = np.sort(df_pos)
    df_neg = dfs['neg_old']

    #neg dfs > this are problematic
    df_thresh = df_pos[.1 * df_pos.shape[0]]

    ind = np.argsort(df)[::-1]

    nnew = len(neg_old) / 2
    nkeep = len(neg_old) - nnew
    nresample = max(nnew, (df > df_thresh).sum())

    old_ind = np.argsort(df_neg)[:nkeep]
    new_ind = np.random.randint(0, nresample - 1, nnew)
    new_ind = ind[new_ind]

    return [neg_old[i] for i in old_ind] + [neg[i] for i in new_ind]
