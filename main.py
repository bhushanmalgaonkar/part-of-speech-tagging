#!/usr/bin/env python3

import math
from time import time

from pos_data import read
from models.simple import Simple
from models.hmm import HMM
from models.complex import Complex
from metrics import print_report


X_train, y_train = read('data/bc.train', 'train')
X_val, y_val = read('data/bc.val', 'train')

# sm = Simple()
# sm.fit(X_train, y_train)
# y_pred = sm.predict(X_val)
# print_report(y_val, y_pred)

# hm = HMM()
# hm.fit(X_train, y_train)
# y_pred = hm.predict(X_val)
# print_report(y_val, y_pred)

for missing_word_cost in [10e-6, 10e-9, 10e-11, 10e-13]:
    for missing_emission_cost in [10e-6, 10e-9, 10e-11, 10e-13]:
        for missing_transition_1_cost in [10e-6, 10e-9, 10e-11, 10e-13]:
            for missing_transition_2_cost in [10e-6, 10e-9, 10e-11, 10e-13]:
                for gibbs in [10, 20, 50, 100, 150]:
                    st = time()

                    cp = Complex()
                    cp.MISSING_WORD_PROBABILITY = missing_word_cost
                    cp.MISSING_WORD_COST = -math.log(missing_word_cost)
                    cp.MISSING_EMISSION_COST = -math.log(missing_emission_cost)
                    cp.MISSING_TRANSITION_1_COST = -math.log(missing_transition_1_cost)
                    cp.MISSING_TRANSITION_2_COST = -math.log(missing_transition_2_cost)
                    cp.NUM_GIBBS_ITER = gibbs

                    cp.fit(X_train, y_train)
                    y_pred = cp.predict(X_val)

                    with open('out.txt', 'a') as f:
                        f.write('{:50s}: {}\n'.format('Missing word', missing_word_cost))
                        f.write('{:50s}: {}\n'.format('Missing emission', missing_emission_cost))
                        f.write('{:50s}: {}\n'.format('Missing transition 1', missing_transition_1_cost))
                        f.write('{:50s}: {}\n'.format('Missing transition 2', missing_transition_2_cost))
                        f.write('{:50s}: {}\n'.format('Gibbs samples', gibbs))

                        f.write(print_report(y_val, y_pred))
                        f.write('{:50s}: {}\n\n'.format('Time taken', int(time() - st)))
