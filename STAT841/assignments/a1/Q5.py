import numpy as np

EPS = 1e-6


def parameters():
    return {
        'W1': np.array([[0.2, -0.1, 0.4, 0.3], [-0.3, 0.5, -0.2, 0.1], [0.1, 0.2, 0.3, -0.4]], dtype=float),
        'b1': np.array([0.1, -0.2, 0.05]),
        'W2': np.array([[0.3, -0.4, 0.2], [-0.1, 0.5, 0.3]], dtype=float),
        'b2': np.array([0.0, 0.1]),
        'v': np.array([0.7, -0.6]),
        'a': np.array([0.2, -0.3, 0.4]),
        'c': np.array(-0.1),
    }


def forward(theta, x, y):
    q1 = theta['W1'] @ x + theta['b1']
    h1 = np.tanh(q1)
    q2 = theta['W2'] @ h1 + theta['b2']
    h2 = np.tanh(q2)
    z = float(theta['v'] @ h2 + theta['a'] @ h1 + theta['c'])
    p = float(np.exp(-np.logaddexp(0.0, -z)))
    loss = float(np.logaddexp(0.0, z) - y * z)
    return p, loss, (h1, h2)


def backward(theta, x, y):
    p, loss, (h1, h2) = forward(theta, x, y)
    dz = p - y
    dq2 = (dz * theta['v']) * (1.0 - h2**2)
    dh1 = theta['W2'].T @ dq2 + dz * theta['a']
    dq1 = dh1 * (1.0 - h1**2)
    grads = {
        'W1': np.outer(dq1, x),
        'b1': dq1,
        'W2': np.outer(dq2, h1),
        'b2': dq2,
        'v': dz * h2,
        'a': dz * h1,
        'c': np.array(dz),
    }
    return p, loss, grads


def finite_difference(theta, x, y, name, index):
    original = float(theta[name][index])
    theta[name][index] = original + EPS
    plus = forward(theta, x, y)[1]
    theta[name][index] = original - EPS
    minus = forward(theta, x, y)[1]
    theta[name][index] = original
    return (plus - minus) / (2.0 * EPS)


def main():
    theta = parameters()
    x = np.array([0.6, -1.2, 0.3, 0.8])
    y = 1
    p, loss, grads = backward(theta, x, y)
    checks = [
        ('(W1)11', 'W1', (0, 0)),
        ('(W1)24', 'W1', (1, 3)),
        ('(b1)3', 'b1', (2,)),
        ('(W2)12', 'W2', (0, 1)),
        ('(W2)23', 'W2', (1, 2)),
        ('(b2)1', 'b2', (0,)),
        ('(v)2', 'v', (1,)),
        ('(a)3', 'a', (2,)),
        ('c', 'c', ()),
    ]
    print(f'p = {p:.12f}')
    print(f'loss = {loss:.12f}')
    print(f'{"coordinate":<12} {"backprop":>17} {"finite difference":>19} {"scaled discrepancy":>20}')
    discrepancies = []
    for label, name, index in checks:
        bp = float(grads[name][index])
        fd = finite_difference(theta, x, y, name, index)
        scaled = abs(bp - fd) / max(1.0, abs(bp), abs(fd))
        discrepancies.append(scaled)
        print(f'{label:<12} {bp:>17.10f} {fd:>19.10f} {scaled:>20.3e}')
    print(f'Maximum scaled discrepancy = {max(discrepancies):.12e}')


if __name__ == '__main__':
    main()