import numpy as np
import pickle
import h5py

arr = np.arange(2**20, dtype=int)
with open("test.b", "wb") as f:
    f.write(arr)

with open("test.pkl", "wb") as f:
    pickle.dump(arr, f)

f = h5py.File("test.h5", "w")
f.create_dataset("ints", data=arr, compression="gzip", compression_opts=9)
f.close()

f = h5py.File("test.h5", "a")
f.create_dataset("ints2", data=arr, compression="gzip", compression_opts=9)
f.close()

exit()
with open("test.b", "rb") as f:
    while True:
        data = f.read(8)
        if not data: break
        print(int.from_bytes(data, "little"))