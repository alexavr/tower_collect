import tower_lib as tl

CONFIG_NAME = "../tower.conf"

config = tl.reader.config(CONFIG_NAME)

dbfile = "%s/%s" % (config['dbpath'], config['dbfile'])
cur = tl.reader.db_init(dbfile)
towers = tl.reader.bd_get_table(cur,"SELECT city,short_name FROM towers")
# df=zip(l1,l2)

print(towers)
print(towers[0])

d = dict(towers)
print(d)