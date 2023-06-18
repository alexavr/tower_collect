# import configparser
# import sys
# from pathlib import Path
# import os
# sys.path.append(Path(os.path.join(os.getcwd(), __file__)).parent.parent.__str__())
import tower_lib as tl
import pandas as pd

# def bd_get_levels(cur, tower):

#     df = pd.DataFrame()

#     print(f"SELECT height FROM equipment WHERE tower_name='{tower}' AND status='online' GROUP BY height ORDER BY height ASC")
#     levels = tl.reader.bd_get_table(cur,f"SELECT height FROM equipment WHERE tower_name='{tower}' AND status='online' GROUP BY height ORDER BY height ASC")
#     tmp_levels = [] 
#     for level in levels:
#         level = level[0]
#         tmp_levels.append(level)
#     df['levels'] = tmp_levels 

#     return df

CONFIG_NAME = "../tower.conf"
config = tl.reader.config(CONFIG_NAME)
dbfile = "%s/%s" % (config['dbpath'], config['dbfile'])
cur = tl.reader.db_init(dbfile)
# towers = tl.reader.bd_get_table(cur,"SELECT short_name FROM towers")
variables = tl.reader.bd_get_levelEqipment(cur, 'PIO', 2)
# print(variables)
levels = tl.reader.bd_get_levels(cur, 'PIO')
# print(levels)

tower_name = 'PIO'
variables = tl.reader.bd_get_table(cur,f"SELECT name FROM variables WHERE tower_name='{tower_name}' ORDER BY height ASC")
# print(variables)

variables = []
# levels = tl.reader.bd_get_table(cur,f"SELECT height FROM equipment \
#     WHERE tower_name='{tower_name}' AND status='online' \
#     GROUP BY height \
#     ORDER BY height ASC")

levels = tl.reader.bd_get_table(cur,f"SELECT height FROM equipment WHERE tower_name='{tower_name}' AND status='online' GROUP BY height ORDER BY height ASC")
# for level in levels:
#     print(level)
#     tmp = tl.reader.bd_get_table(cur,f"SELECT name FROM variables \
#         WHERE tower_name='{tower_name}' AND height='{level[0]}' AND status='online' \
#         ORDER BY height ASC")
#     tmp = list(tmp)
#     tmp = [tpl[0] for tpl in tmp]
#     tmp = ', '.join(tmp)
#     # print(tmp)
#     # variables.append(tmp)

equipment = []
for level in levels:
    tmp = tl.reader.bd_get_table(cur,f"SELECT name FROM equipment WHERE tower_name='{tower_name}' AND height='{level[0]}' AND status='online' ORDER BY height ASC")
    print(level,tmp)
    tmp = [tpl[0] for tpl in tmp]
    print(level,tmp)
    tmp = ', '.join(tmp)
    print(level,tmp)
    equipment.append(tmp)
print(equipment)
    # equipment = []
    # for level in levels:
    #     tmp = tl.reader.bd_get_table(cur,f"SELECT name FROM equipment WHERE tower_name='{tower_name}' AND height='{level[0]}' AND status='online' GROUP BY height ORDER BY height ASC")
    #     tmp = list(tmp)
    #     tmp = [tpl[0] for tpl in tmp]
    #     tmp = ', '.join(tmp)
    #     equipment.append(tmp)

# levels = tl.reader.bd_get_table(cur,f"SELECT name,height FROM equipment WHERE tower_name='{tower_name}' AND status='online' GROUP BY height ORDER BY height ASC")
# print(levels)

# print(variables)

# tower_name = 'PIO'
# print(tl.reader.bd_get_levels(cur, tower_name))
# print(tl.reader.bd_get_table(cur,f"SELECT height FROM equipment WHERE tower_name='{tower_name}' AND status='online' GROUP BY height ORDER BY height ASC"))


# def get_towers(cur):
#     import sqlite3
    
#     result = cur.execute('SELECT short_name FROM towers ')

#     towers = []
#     for tower in result:
#         towers.append(tower)

#     return towers

# def get_levels(cur, tower_name):
#     import sqlite3
   
#     result = cur.execute("SELECT height FROM equipment WHERE tower_name=? AND status='online' GROUP BY height ORDER BY height ASC", (tower_name,)) # DESC, ASC

#     levels = []
#     for level in result:
#         levels.append(level[0])

#     return levels

# def get_variables(cur, tower_name, height):
#     import sqlite3
   
#     result = cur.execute('SELECT name FROM variables WHERE tower_name=? AND height=? ORDER BY height ASC', (tower_name,height))

#     variables = []
#     for variable in result:
#         variables.append(variable[0])

#     return variables





