import tkinter as tk
import tkinter.ttk as ttk

class Input():
    _type = {
        str: tk.StringVar,
        int: tk.IntVar,
        float: tk.DoubleVar,
        bool: tk.BooleanVar
    }
    def __init__(self, tag, disp, typ, opts=None, adj=None):
        self.tag = tag
        self.disp = disp
        self.ref = self._type[typ]()
        self.opts = opts
        self.adj = adj

    def make_label(self, master):
        return ttk.Label(master=master, text=self.disp)
    
    def make_entry(self, master):
        if self.opts is None:
            return ttk.Entry(master=master, textvariable=self.ref)
        else:
            temp = ttk.Combobox(master=master, textvariable=self.ref)
            temp["values"] = self.opts
            return temp

config = {
    "control": {
        "name": "ethylene",
        "tunit": "au",
        "tmax": 5000,
        "dt": 10,
        "nqsteps": 20,
    },
    "nuclear": {
        "input": "geom.xyz",
        "nucupd": "vv",
    },
    "electronic": {
        "program": "molpro",
        "path": "",
        "type": "cas",
        "states": [2,0,0],
        "options": {
            "basis": "6-31g**",
            "closed": 7, 
            "active": 9,
            "nel": 16,
            "sa": 3,
            "mld": False,
            "df": False
        },
        "tdcupd": "nacme",
        "cupd": "propmat",
    },
    "dynamics": {
        "method": "hop",
        "type": "fssh",
        "decoherence": "none",
        "initstate": 1,
    },
    "output": {
        "file": "out",
        "record": ["pop", "pes", "pen", "ken", "en"],
    }
}

def populate_tab(master, content: dict[str, Input]):
    for i, (key, val) in enumerate(content.items()):
        val.make_label(master).grid(row=i, column=0, **pad_in)
        val.make_entry(master).grid(row=i, column=1, **pad_in)

pad_in = {
    "padx": 2,
    "pady": 2,
}

root = tk.Tk()
ntb = ttk.Notebook(root)
ntb.grid(padx=5, pady=5)


ctrl = {
    "name": Input("name", "System name", str),
    "tunit": Input("tunit", "Time units", str),
    "dt": Input("dt", "Time step", float),
    "tmax": Input("tmax", "End time", float),
    "nqsteps": Input("nqsteps", "Quantum steps", int)
}
def generate_control():
    frm_ctrl = ttk.Frame(ntb)
    ntb.add(frm_ctrl, text="Control")
    populate_tab(frm_ctrl, ctrl)


nuc = {
    "input": Input("input", "Geometry", str),
    "nucupd": Input("nucupd", "Integrator", str, opts=[
        "Velocity Verlet",
        "Symmetric Multistep",
        "Runge-Kutta-Nystrom"
    ])
}
def generate_nuc():
    frm_nuc = ttk.Frame(ntb)
    ntb.add(frm_nuc, text="Nuclear")
    populate_tab(frm_nuc, nuc)

generate_control()
generate_nuc()

frm_elc = ttk.Frame(ntb)
ntb.add(frm_elc, text="Electronic")
root.mainloop()