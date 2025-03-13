

"""
Input file:
geom <xyz>
states <n>
grad <i>
...
nac <i> <j>


Command:
python3 $SH/electronic/run_demo.py <input>


Outputs:
ham.npy
grad.npy
nac.npy

"""

def spring(self):
    with open("input.inp", "w") as f:
        f.write(f"geom {self._file}.xyz\n")

        f.write(f"states {self.n_states}\n")

        for i in range(self.n_states):
            if self._calc_grad[i]:
                f.write(f"grad {i}\n")

        for i in range(self.n_states):
            for j in range(i):
                if self._calc_nac[i,j]:
                    f.write(f"nac {j} {i}\n")
