import json
from database import Database
db = Database('BATCH01', 'GC_QA')
data = db.fetchall()
