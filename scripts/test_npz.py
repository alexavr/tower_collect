import numpy as np
from datetime import datetime

b = np.load('../data/buffer/MSU_A1_BUFFER.npz')

print(b.files)
# print(b['time'])

# timedata1 = datetime.fromtimestamp(b['time'][0]).strftime('%Y-%m-%d %H:%M:%S')
# timedata2 = datetime.fromtimestamp(b['time'][-1]).strftime('%Y-%m-%d %H:%M:%S')
print(f"Time: {datetime.fromtimestamp(b['time'][0]).strftime('%Y-%m-%d %H:%M:%S')} ... {datetime.fromtimestamp(b['time'][-1]).strftime('%Y-%m-%d %H:%M:%S')}")

