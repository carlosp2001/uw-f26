import numpy as np

def ReadX(path):
    print(f'>>> Reading data from: {path} ...')
    with open(path) as f:
        # only one line that includes everything
        file = f.readlines()

    print(f'#instances: {len(file)}') # 7352 for training set, 2947 for test set

    X_all = []
    for instance in file:
        f = filter(None, instance.split(' '))
        instance_filterd = list(f)
        instance_cleaned = [float(attr.strip()) for attr in instance_filterd]
        X_all.append(instance_cleaned)
    X_all = np.array(X_all)
    print('>>> Reading finished! Data are converted to numpy array.')
    print(f'shape of X: {X_all.shape} ==> each instance has {X_all.shape[1]} attributes.')

    return X_all

def ReadY(path):
    print(f'>>> Reading data from: {path} ...')
    with open(path) as f:
        # only one line that includes everything
        file = f.readlines()

        print(f'#instances: {len(file)}')  # 7352 for training set, 2947 for test set

    y_all = [float(label.strip()) for label in file]
    y_all = np.array(y_all)
    print('>>> Reading finished! Data are converted to numpy array.')
    print(f'shape of y: {y_all.shape}')
    return y_all