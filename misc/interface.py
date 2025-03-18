import curses
from curses import wrapper
import curses.textpad
import sys, os
import pickle, json
import time

from classes.constants import is_true

FG = {
    -1: "\033[0m",
    0:  "\033[31m",
    1:  "\033[33m",
    2:  "\033[32m",
    3:  "\033[30m"
}

BG = {
    -1: "\033[0m",
    0:  "\033[41m",
    1:  "\033[43m",
    2:  "\033[42m",
    3:  "\033[47m"
}

DEFAULT = {
    "control": {
        "text": "General",
        "children": {
            "trajtype": {
                "text": "Trajectory type",
                "default": "sh",
                "type": str,
                "help": f'''
Available trajectory types
{FG[3]}{BG[3]}sh{FG[-1]}: surface hopping
{FG[3]}{BG[3]}mfe{FG[-1]}: mean-field Ehrenfest'''
            },
            "location": {
                "text": "Path to ensemble",
                "default": ".",
                "type": str
            },
            "name": {
                "text": "Name of molecule",
                "default": "x",
                "type": str
            },
            "record": {
                "text": "Quantities to record",
                "default": "pes",
                "type": str,
                "list": True
            },
            "tunit": {
                "text": "Input units",
                "default": "au",
                "type": str
            },
            "tmax": {
                "text": "Total time",
                "default": None,
                "type": int
            },
            "adapt": {
                "text": "Adaptive stepsize",
                "default": "n",
                "type": bool
            },
            "step": {
                "text": "Stepsize",
                "default": None,
                "type": float
            },
            "stepmax": {
                "text": "Max stepsize",
                "default": None,
                "type": float
            },
            "stepmin": {
                "text": "Min stepsize",
                "default": None,
                "type": float
            },
            "stepfunc": {
                "text": "Stepsize function",
                "default": "tanh",
                "type": str
            },
            "stepvar": {
                "text": "Stepsize variable",
                "default": "nac**2",
                "type": str
            },
            "stepparams": {
                "text": "Stepsize parameters",
                "default": None,
                "type": float,
                "list": True
            },
            "qres": {
                "text": "Number of quantum substeps",
                "default": "20",
                "type": int
            },
            "enthresh": {
                "text": "Energy thresholds",
                "default": None,
                "type": float,
                "list": True
            }
        }
    },
    "nuclear": {
        "text": "Nuclear",
        "children": {
            "format": {
                "text": "Input format",
                "default": "xyz",
                "type": str
            },
            "integrator": {
                "text": "Nuclear integrator",
                "default": "vv",
                "type": str
            }
        }
    },
    "electronic": {
        "text": "Electronic",
        "children": {
            "nstates": {
                "text": "Number of states (S D T)",
                "default": None,
                "type": int,
                "list": True
            },
            "initstate": {
                "text": "Initial state (0-indexed)",
                "default": None,
                "type": int
            },
            "skip": {
                "text": "Skip states",
                "default": "0",
                "type": int
            },
            "propagator": {
                "text": "WF propagator",
                "default": "propmat",
                "type": str
            },
            "program": {
                "text": "Program",
                "default": "molpro",
                "type": str
            },
            "programpath": {
                "text": "Path to program",
                "default": None,
                "type": str
            },
            "esttype": {
                "text": "EST calculation type",
                "default": "casscf",
                "type": str
            },
            "tdc": {
                "text": "TDC method",
                "default": "npi",
                "type": str
            },
            "config": {
                "text": "Config",
                "children": {
                    "nel": {
                        "text": "Number of electrons",
                        "default": None,
                        "type": int
                    },
                    "closed": {
                        "text": "Closed orbitals",
                        "default": None,
                        "type": int
                    },
                    "active": {
                        "text": "Active orbitals",
                        "default": None,
                        "type": int
                    },
                    "basis": {
                        "text": "Basis",
                        "default": None,
                        "type": str
                    },
                    "sa": {
                        "text": "State average",
                        "default": None,
                        "type": int
                    },
                    "df": {
                        "text": "Density fitting",
                        "default": "n",
                        "type": bool
                    },
                    "dfbasis": {
                        "text": "Basis for density fitting",
                        "default": None,
                        "type": str
                    },
                    "mld": {
                        "text": "Write molden",
                        "default": None,
                        "type": int
                    },
                    "mstype": {
                        "text": "MS-CASPT2 type",
                        "default": "X",
                        "type": str
                    },
                    "imag": {
                        "text": "Imaginary shift",
                        "default": "0",
                        "type": float
                    },
                    "shift": {
                        "text": "Real shift",
                        "default": "0",
                        "type": float
                    },
                    "ipea": {
                        "text": "IPEA shift",
                        "default": "0",
                        "type": float
                    },
                    "sig2": {
                        "text": "Sig2 renormalisation",
                        "default": "0",
                        "type": float
                    },
                }
            }
        }
    },
    "hopping": {
        "text": "Surface Hopping",
        "children": {
            "shtype": {
                "text": "Hopping type",
                "default": "fssh",
                "type": str
            },
            "decoherence": {
                "text": "Decoherence",
                "default": "edc",
                "type": str
            }

        }
    }
}

adaptive = {
    "on": [
        "stepfunc",
        "stepvar",
        "stepmax",
        "stepmin",
        "stepparams",
    ],
    "off": [
        "step",
    ]
}

sh = {
    "on": [
        "hopping"
    ],
    "off": [

    ]
}

df = {
    "on": [
        "dfbasis"
    ],
    "off": []
}

pt2 = {
    "on": [
        "mstype",
        "imag",
        "shift",
        "ipea",
        "sig2"
    ],
    "off": []
}

class Box():
    def __init__(self, scr, menubox, inpheaderbox, inpbox, hlpbox):
        self.scr = scr
        self.menu = menubox
        self.inph = inpheaderbox
        self.inp = inpbox
        self.hlp = hlpbox


class Menu():
    def __init__(self, text):
        self.text = text
        self.hlp = ""
        self.children: list[Selectable | TextField] = []
        self.vis_children: list[Selectable | TextField] = []
        self.selected_child = None
        self.status = 0
        self.modified = False

    def construct(self, tree: dict):
        for key, val in tree.items():
            default = val.get("default", "")
            text = val.get("text")
            if default: text += f" [{default}]"
            child = Selectable(key, text, val.get("help", "No help available"))
            if children := val.get("children"):
                child.construct(children)
            else:
                child.append_child(TextField(text=default, tp=val.get("type"), lst=val.get("list", False)))
            self.append_child(child)
        self.vis_children = self.children

    def display(self):
        self.get_status()

        box.menu.clear()
        box.menu.addstr(0, 0, self.text, curses.A_UNDERLINE)
        box.menu.addstr(1, 0, "Navigation: arrow keys; Exit: Ctrl + C; Save: Ctrl + S.")
        self.alter_visibility(adaptive["on"], adaptive["off"], is_true(self.get_child_by_tag("adapt").children[0].text))
        self.alter_visibility(sh["on"], sh["off"], "sh" in self.get_child_by_tag("trajtype").children[0].text)
        self.alter_visibility(df["on"], df["off"], is_true(self.get_child_by_tag("df").children[0].text))
        self.alter_visibility(pt2["on"], pt2["off"], "caspt2" in self.get_child_by_tag("esttype").children[0].text.lower())
        self.display_children(0)
        box.menu.refresh()

    def display_children(self, depth: int):
        for i, child in enumerate(self.vis_children):
            if child == self.selected_child:
                box.menu.addstr(i+2, 40*depth, child.text, STAT[child.status] | curses.A_REVERSE)
                if child.selected_child is not None and not isinstance(child.selected_child, TextField):
                    child.display_children(depth+1)
            else:
                box.menu.addstr(i+2, 40*depth, child.text, STAT[child.status])

    def display_help(self):
        box.hlp.clear()
        lines = [s for s in self.hlp.split("\n") if s]
        box.hlp.addstr(0, 0, "Help", curses.A_UNDERLINE)
        for i, line in enumerate(lines):
            box.hlp.addstr(1 + i, 0, line)
        box.hlp.refresh()

    def display_input(self):
        box.inph.clear()
        box.inph.refresh()

        box.inp.clear()
        box.inp.refresh()


    def alter_visibility(self, on: list, off: list, orig: bool):
        for name in on:
            self.get_child_by_tag(name).visible = orig
        for name in off:
            self.get_child_by_tag(name).visible = not orig

    def append_child(self, child):
        self.children.append(child)
        child.parent = self
        return self

    def get_child_by_text(self, text: str):
        for child in self.children:
            if isinstance(child, TextField): continue

            if text == child.text:
                return child

            if (temp := child.get_child_by_text(text)) is None: continue
            else: return temp

    def get_child_by_tag(self, text: str):
        for child in self.children:
            if isinstance(child, TextField): continue

            if text == child.tag:
                return child

            if (temp := child.get_child_by_tag(text)) is None: continue
            else: return temp

    def remove_child(self, child):
        for i, other in enumerate(self.children):
            if other == child:
                self.children.pop(i)
                break
        del child

    def next_child(self):
        for i, other in enumerate(self.vis_children):
            if other == self.selected_child:
                self.selected_child = self.vis_children[(i+1)%len(self.vis_children)]
                return

    def prev_child(self):
        for i, other in enumerate(self.vis_children):
            if other == self.selected_child:
                self.selected_child = self.vis_children[(i-1)%len(self.vis_children)]
                return

    def deselect_children(self):
        self.selected_child = None
        for child in self.children:
            if isinstance(child, TextField):
                return
            else:
                child.deselect_children()

    def get_status(self):
        self.vis_children = [c for c in self.children if c.visible]
        stat = 2
        for child in self.vis_children:
            stat = min(child.get_status(), stat)

        self.modified = self.status != stat
        self.status = stat
        return self.status

    def to_dict(self):
        d = {}
        for child in self.vis_children:
            if isinstance(child, TextField):
                return child.convert()
            else:
                temp = child.to_dict()
                d[child.tag] = temp
        return d

class Selectable(Menu):
    def __init__(self, tag: str, text: str, hlp: str):
        self.tag = tag
        self.text = text
        self.hlp = hlp
        self.parent: Selectable | Menu = None
        self.children: list[Selectable | TextField] = []
        self.vis_children: list[Selectable | TextField] = []
        self.selected_child = None
        self.check = False
        self.parent: Selectable | Menu = None
        self.status = 0
        self.visible = True
        self.modified = False

class TextField():
    def __init__(self, text: str = "", tp: type = str, lst: bool = False):
        self.text = text or ""
        self.vals = text
        self.parent: Selectable = None
        self.check_text = None
        self.visible = True
        self.status = bool(text)
        self.input_type = tp
        self.lst = lst

    def convert(self):
        def cnv(inp: str):
            if self.input_type == bool:
                return is_true(inp)
            else:
                return self.input_type(inp)

        if self.lst:
            return [cnv(i) for i in self.text.replace("\n", " ").split()]
        else:
            out = "\n".join(self.text.replace("\n", " ").split())
            return cnv(out)

    def check_input(self):
        try:
            inp = self.convert()
            if inp: return 2
            else: return 0
        except:
            return 0

    def display_input(self):
        box.inph.clear()
        box.inph.addstr(0, 0, "Input", curses.A_UNDERLINE)
        box.inph.refresh()

        box.inp.clear()
        box.inp.addstr(0, 0, self.text)
        box.inp.refresh()

    def get_input(self):
        self.display_input()

        txt = curses.textpad.Textbox(box.inp)
        curses.curs_set(1)
        txt.edit()

        self.text = txt.gather()
        curses.curs_set(0)

        self.status = self.check_input()


    def get_status(self):
        return self.status

def merge(into: dict, ref: dict):
    for key, val in ref.items():
        if not key in into.keys():
            into[key] = val
        elif isinstance(val, dict):
            into[key] = merge(into[key], ref[key])
    return into

'''
\u2554\u2550\u2557
\u2551      \u2551
\u2560\u2550\u2563
\u2551      \u2551
\u255a\u2550\u255d
'''

def rectangle(win, uly, ulx, lry, lrx):
    """Draw a rectangle with corners at the provided upper-left
    and lower-right coordinates.
    """
    win.vline(uly+1, ulx, curses.ACS_VLINE, lry - uly - 1)
    win.hline(uly, ulx, curses.ACS_HLINE, lrx - ulx - 1)
    win.hline(lry, ulx, curses.ACS_HLINE, lrx - ulx - 1)
    win.vline(uly+1, lrx, curses.ACS_VLINE, lry - uly - 1)
    win.insch(uly, ulx, curses.ACS_ULCORNER)
    win.insch(uly, lrx, curses.ACS_URCORNER)
    win.insch(lry, ulx, curses.ACS_LLCORNER)
    win.insch(lry, lrx, curses.ACS_LRCORNER)

def main(stdscr):
    if len(sys.argv) > 1:
        inp = sys.argv[1]
        if inp.endswith(".pkl"):
            with open(inp, "rb") as f:
                menu: Menu = pickle.load(f)

        elif inp.endswith(".json"):
            with open(sys.argv[1], "r") as f: tree = json.load(f)
            tree = merge(tree, DEFAULT)
            menu = None

        else:
            print("Error: Unexpected input file type")
            exit(0)
    else:
        tree = DEFAULT
        menu = None

    curses.init_pair(1, curses.COLOR_RED, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_GREEN, curses.COLOR_BLACK)

    global STAT
    STAT = {
        0: curses.color_pair(1),
        1: curses.color_pair(2),
        2: curses.color_pair(3)
    }

    global box
    box = Box(
        stdscr,
        curses.newwin(curses.LINES//2-2, curses.COLS-2, 1, 1),
        curses.newwin(1, curses.COLS//2-2, curses.LINES//2+1, 1),
        curses.newwin(curses.LINES-curses.LINES//2-3, curses.COLS//2-2, curses.LINES//2+2, 1),
        curses.newwin(curses.LINES-curses.LINES//2-2, curses.COLS-curses.COLS//2-3, curses.LINES//2+1, curses.COLS//2+2))

    curses.curs_set(0)
    rectangle(box.scr, 0, 0, curses.LINES//2-1, curses.COLS-1)
    rectangle(box.scr, curses.LINES//2, 0, curses.LINES-1, curses.COLS//2-1)
    rectangle(box.scr, curses.LINES//2, curses.COLS//2, curses.LINES-1, curses.COLS-1)
    box.scr.refresh()

    if menu is None:
        menu = Menu("Selection Menu")
        menu.construct(tree)
    else:
        menu.deselect_children()
    menu.selected_child = menu.children[0]
    active = menu
    active.selected_child.display_help()

    menu.display()
    active.selected_child.display_help()

    while True:
        key = box.scr.getkey()
        if key == "q":
            exit()

        elif key == "d":
            menu.alter_visibility(adaptive, False)

        elif key == "h":
            continue

        elif key == "s":
            with open("save.pkl", "wb") as f:
                pickle.dump(menu, f)

            if menu.status > 0:
                with open("input.json", "w") as f:
                    json.dump(menu.to_dict(), f, indent=4)
            continue

        # arrows
        elif key == "KEY_UP":
            active.prev_child()
            if isinstance(active.selected_child.children[0], TextField):
                active.selected_child.children[0].display_input()
        elif key == "KEY_DOWN":
            active.next_child()
            if isinstance(active.selected_child.children[0], TextField):
                active.selected_child.children[0].display_input()
        elif key == "KEY_RIGHT":
            active = active.selected_child
            if isinstance(active.children[0], TextField):
                active.children[0].get_input()
                menu.display()
                active = active.parent
            else:
                active.selected_child = active.vis_children[0]
                if isinstance(active.selected_child.children[0], TextField):
                    active.selected_child.children[0].display_input()
        elif key == "KEY_LEFT" and active != menu:
            active.selected_child = None
            active = active.parent
        menu.display()
        active.selected_child.display_help()

if __name__ == "__main__":
    wrapper(main)