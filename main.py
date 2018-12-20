#!/usr/bin/env python3

from pos_data import read
from models.simple import Simple
from models.hmm import HMM
from models.complex import Complex
from metrics import print_report


X_train, y_train = read('data/bc.train', 'train')
X_val, y_val = read('data/bc.val', 'train')

print('Simple model')
sm = Simple()
sm.fit(X_train, y_train)
y_pred = sm.predict(X_val)
print_report(y_val, y_pred)

print('HMM')
hm = HMM()
hm.fit(X_train, y_train)
y_pred = hm.predict(X_val)
print_report(y_val, y_pred)

print('Complex model')
cp = Complex()
cp.fit(X_train, y_train)
y_pred = cp.predict(X_val)
print_report(y_val, y_pred)
