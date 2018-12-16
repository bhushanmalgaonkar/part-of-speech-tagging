from pos_data import read
from simple import Simple
from metrics import print_report


X_train, y_train = read('data/bc.train')
X_val, y_val = read('data/bc.test')

sm = Simple()
sm.fit(X_train, y_train)
y_pred = sm.predict(X_val)

print_report(y_val, y_pred)
