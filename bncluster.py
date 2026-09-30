import numpy as np
from sklearn.cluster import KMeans

def BNcluster(args, x_np, y_np):
    mask0 = (y_np == 0)
    x_np0 = x_np[mask0]

    mask_mal = (y_np > 0)
    y_np_mal = y_np[mask_mal]

    x_np0_flat = x_np0.reshape(x_np0.shape[0], -1)
    km = KMeans(n_clusters=args.bn_cluster, random_state=0)
    y_np0_new = km.fit_predict(x_np0_flat)
    clucnt = np.bincount(y_np0_new, minlength=args.bn_cluster)

    y_np_mal_new = y_np_mal + args.bn_cluster - 1

    y_np_new = np.zeros_like(y_np)
    y_np_new[mask0] = y_np0_new
    y_np_new[mask_mal] = y_np_mal_new

    return y_np_new
