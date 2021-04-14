import numpy as np

def blobs_old(n_samples, n_features, centers=[0,0], cluster_std=[1,1], random_state=1):
    n = int(np.floor(n_samples/2))
    rng = np.random.default_rng(random_state)
    
    if len(centers) == 2:
        class1 = rng.normal(centers[0], cluster_std[0]**0.5, (n,n_features))
        class2 = rng.normal(centers[1], cluster_std[1]**0.5, (n,n_features))
        X = np.vstack((class1,class2))
        y = np.hstack((np.repeat(0, n).flatten(), np.repeat(1, n).flatten()))
    elif len(centers) == 1:
        class1 = rng.normal(centers[0], cluster_std[0]**0.5, (n,n_features))
        X = class1
        y = np.repeat(0, n).flatten()   
        
    idx = np.random.permutation(range(X.shape[0]))
    
    return X[idx], y[idx]


def blobs(n_samples, n_features, center1=0, center2=None, cluster_std=[1,1], random_state=1):
    n = int(np.floor(n_samples/2))
    rng = np.random.default_rng(random_state)
    
    e1 = np.zeros(n_features)
    e2 = np.zeros(n_features)
    e1[0] = center1
    e2[0] = center2
    
    
    if center2 is not None:
        class1 = rng.normal(e1, cluster_std[0]**0.5, (n,n_features))
        class2 = rng.normal(e2, cluster_std[1]**0.5, (n,n_features))
        X = np.vstack((class1,class2))
        y = np.hstack((np.repeat(0, n).flatten(), np.repeat(1, n).flatten()))
    elif center2 is None:
        class1 = rng.normal(e1, cluster_std[0]**0.5, (n,n_features))
        X = class1
        y = np.repeat(0, n).flatten()   
        
    idx = np.random.permutation(range(X.shape[0]))
    
    return X[idx], y[idx]





def cube(n_samples, n_features, random_state=1):
    rng = np.random.default_rng(random_state)
    X = rng.uniform(low=0.0, high=1.0, size=(n_samples,n_features))
    y = np.asarray([1 if X[i,0] >0.5 else 0 for i in range(n_samples)])
    idx = np.random.permutation(range(X.shape[0]))
    return X[idx], y[idx]

def cube(n_samples, n_features, random_state=1):
    rng = np.random.default_rng(random_state)
    X = rng.uniform(low=0.0, high=1.0, size=(n_samples,n_features))
    y = np.asarray([1 if X[i,0] >0.5 else 0 for i in range(n_samples)])
    X[y==1] *= 1.2
    idx = np.random.permutation(range(X.shape[0]))
    return X[idx], y[idx]

def CommonCenterClassifier(X_train, X_test):
    
    # Learn a Threshold using training data
    threshold = np.median(np.linalg.norm(X_train, axis=1))
    
    # Classify test data
    preds = [0 if np.linalg.norm(X_test[i,:]) < threshold else 1 for i in range(X_test.shape[0])]  
    
    # Return predictions
    return preds
