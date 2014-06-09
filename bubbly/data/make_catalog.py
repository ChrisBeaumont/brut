from MySQLdb import connect
import numpy as np
import cPickle as pickle

pth = 'catalog.pkl'

db = connect(host='localhost', user='beaumont', db='mwp')
cursor = db.cursor()
cursor.execute('select lon, lat, angle, semi_major, semi_minor, '
               'thickness, hit_rate from clean_bubbles_anna')
cat = np.array([map(float, row) for row in cursor.fetchall()])

with open(pth, 'w') as outfile:
    pickle.dump(cat, outfile)
