import numpy as np
from dsacv02.tools import smoothing


if __name__ == '__main__':
    # Recommended Smoothing: 0.85
    ara = np.arange(start=1, stop=101, step=1, dtype=np.float64)
    ara_complex = np.where(ara % 2 == 0, ara, -ara)
    ones = np.ones((10000,), dtype=np.float64)

    weight_r1 = np.array(0.1, dtype=np.float64)
    weight_r2 = np.array(0.85, dtype=np.float64)

    ara_sm_r1 = smoothing(scalars=ara, weight=weight_r1)
    ara_complex_sm_r1 = smoothing(scalars=ara_complex, weight=weight_r1)
    ones_sm_r1 = smoothing(scalars=ones, weight=weight_r1)

    ara_sm_r2 = smoothing(scalars=ara, weight=weight_r2)
    ara_complex_sm_r2 = smoothing(scalars=ara_complex, weight=weight_r2)
    ones_sm_r2 = smoothing(scalars=ones, weight=weight_r2)

    print(f'Smoothed arange with weight {weight_r1} array is: {ara_sm_r1}')
    print(f'Smoothed arange with weight {weight_r2} array is: {ara_sm_r2}')
    print('')
    print(f'Smoothed arange complex with weight {weight_r1} array is: {ara_complex_sm_r1}')
    print(f'Smoothed arange complex with weight {weight_r2} array is: {ara_complex_sm_r2}')

    # print(f'Smoothed ones with weight {weight_r1}array is: {ones_sm_r1}')
    # print(f'Smoothed ones with weight {weight_r2}array is: {ones_sm_r2}')

    # Test piece-wise calculation
    ara_batches = ara.reshape(10, 10)
    weight = 0.85
    total = list()
    for idx, batch in enumerate(ara_batches):
        if idx == 0:
            last = 0
            iter_n = 0
        batch_sm, iter_n, last = smoothing(scalars=batch, weight=weight, iter=iter_n, last=last)
        total.append(batch_sm)
    print(f'Piece-wise smoothed with weight {weight} is: {total}')

