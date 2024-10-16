from NssMPC import ArithmeticSecretSharing


def reciprocal(x):
    result = 3 * ArithmeticSecretSharing.exp(1 - x * 2) + 0.003
    for i in range(10):
        result = 2 * result - result * result * x
    return result


def crypten_div(x, y):
    return x * reciprocal(y)
