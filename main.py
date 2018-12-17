from pos_data import read
from models.simple import Simple
from metrics import print_report


X_train, y_train = read('data/bc.train', 'train')
X_val, y_val = read('data/bc.val', 'train')

sm = Simple()
sm.fit(X_train, y_train)
y_pred = sm.predict(X_val)
