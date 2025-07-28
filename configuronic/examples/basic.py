import numpy as np
import configuronic as cfn


def noisy_sin(w, th, noise_std=0.1):
    t = np.linspace(0, 1, 100)
    return np.sin(w * t + th) + np.random.normal(0, noise_std, 100)


noisy_sin_01 = cfn.Config(noisy_sin, w=1, th=0)
clean_sin = noisy_sin_01.override(noise_std=0)


@cfn.config()
def second_order_polynomial(a=1, b=0, c=0):
    x = np.linspace(0, 1, 100)
    return a * x**2 + b * x + c


def print_exp_moving_average(sequence, alpha=0.1):
    result = None
    for x in sequence:
        result = x if result is None else alpha * x + (1 - alpha) * result
        print(result)
    return result


main = cfn.Config(print_exp_moving_average, sequence=noisy_sin_01)

if __name__ == "__main__":
    cfn.cli(main)
