# save the database content to variable db
# should be used in ipython as %run open_final.py

import sys; sys.path.append('..')
from load_save import read_final
db = read_final()
