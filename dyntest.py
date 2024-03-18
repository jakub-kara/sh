import sys, os
import termios, tty
import json, pickle
import string

from constants import Constants

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

MENUHDR = (1, 1, os.get_terminal_size()[0]-1, 3)
MENUBOX = (1, 3, os.get_terminal_size()[0]-1, 19)
INPUTBOX = (1, 20, os.get_terminal_size()[0]//2, os.get_terminal_size()[1]-1)
HELPBOX = (os.get_terminal_size()[0]-os.get_terminal_size()[0]//2+1, 20, os.get_terminal_size()[0]-1, os.get_terminal_size()[1]-1)

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
        clear_block(*MENUBOX)
        self.get_status()

        show(f"\033[4m{self.text}\033[0m", 1, 1)
        show("Navigation: arrow keys; Exit: Ctrl + C; Save: Ctrl + S.", 1, 2)
        self.alter_visibility(adaptive["on"], adaptive["off"], self.get_child_by_tag("adapt").children[0].lines[0] in Constants.true)
        self.alter_visibility(sh["on"], sh["off"], self.get_child_by_tag("trajtype").children[0].lines[0] == "sh")
        self.alter_visibility(df["on"], df["off"], self.get_child_by_tag("df").children[0].lines[0] in Constants.true)
        self.alter_visibility(pt2["on"], pt2["off"], self.get_child_by_tag("esttype").children[0].lines[0].lower() == "caspt2")

        self.display_children(0)

    def display_children(self, depth: int):
        for i, child in enumerate(self.vis_children):
            set_position(40*depth+1, i+3)
            if child == self.selected_child:
                print(f"\033[30m{BG[child.status]}{child.text}\033[0m")
                if child.selected_child is not None and not isinstance(child.selected_child, TextField):
                    child.display_children(depth+1)
            else:
                print(f"{FG[child.status]}{child.text}{FG[-1]}")
    
    def display_help(self):
        lines = [s for s in self.hlp.split("\n") if s]
        show("\033[4mHelp\033[0m", HELPBOX[0], HELPBOX[1])
        for i, line in enumerate(lines):
            show(line, HELPBOX[0], HELPBOX[1]+i+1)
    
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
        self.lines = [text or ""]
        self.parent: Selectable = None
        self.y = len(self.lines) - 1
        self.x = len(self.lines[self.y])
        self.check_text = None
        self.visible = True
        self.status = bool(text)
        self.input_type = tp
        self.lst = lst

    def update_input(self, redraw=[]):
        for i in redraw:
            clear_block(1, 21+i, INPUTBOX[2]-INPUTBOX[0]+1, 22+i)
            if i < len(self.lines): show(self.lines[i], 1, 21+i)
        set_position(self.x+1, self.y + 21)

    def convert(self):
        def cnv(inp: str):
            if self.input_type == bool: return inp in Constants.true
            else: return self.input_type(inp)

        if self.lst:
            out = " ".join(self.lines)
            return [cnv(i) for i in out.split()]
        else:
            out = "\n".join(self.lines)
            return cnv(out)
    
    def check_input(self):
        try:
            inp = self.convert()
            if inp: return 2
            else: return 0
        except:
            return 0

    def get_input(self):
        show("\033[4mInput\033[0m", 1, 20)
        self.update_input([i for i in range(len(self.lines))])

        redraw = []
        while True:
            self.update_input(redraw)
            redraw = []
            
            char = getchar()

            # Ctrl + S
            if ord(char) == 19:
                clear_block(*INPUTBOX)
                self.status = self.check_input()
                break

            # enter
            if ord(char) == 13:
                if self.y + 2 >= INPUTBOX[3] - INPUTBOX[1]: continue
                temp = self.lines[self.y][self.x:]
                self.lines[self.y] = self.lines[self.y][:self.x]
                redraw = [i for i in range(self.y, len(self.lines))]
                self.lines.insert(self.y+1, temp)
                if temp: redraw.append(self.y + 1)
                self.y += 1
                self.x = 0
                continue

            # backspace
            """if ord(char) == 8:
                if self.x == 0:
                    if self.y > 0:
                        redraw = [i for i in range(self.y-1, len(self.lines))]
                        self.y -= 1
                        self.x = len(self.lines[self.y])
                        self.lines[self.y] += self.lines[self.y + 1]
                        self.lines.pop(self.y + 1)
                else:
                    redraw.append(self.y)
                    self.lines[self.y] = self.lines[self.y][:self.x-1] + self.lines[self.y][self.x:]
                    self.x -= 1
                    print("\033[D", end="", flush=True)
                continue"""
            if ord(char) == 8:
                if self.x == 0:
                    if self.y > 0 and len(self.lines[self.y]) == 0:
                        redraw = [i for i in range(self.y-1, len(self.lines))]
                        self.y -= 1
                        self.x = len(self.lines[self.y])
                        self.lines.pop(self.y + 1)
                else:
                    redraw.append(self.y)
                    self.lines[self.y] = self.lines[self.y][:self.x-1] + self.lines[self.y][self.x:]
                    self.x -= 1
                    print("\033[D", end="", flush=True)
                continue


            # escape sequence
            if ord(char) == 27:
                while True:
                    seq = getchar()
                    if ord(seq) == 27: continue
                    if seq == "q": exit()
                    if seq.isalpha() or seq == "~":
                        break

                if seq == "A" and self.y > 0:
                    self.y -= 1
                    print("\033[A", end="", flush=True)
                    if self.x > len(self.lines[self.y]):
                        self.x = len(self.lines[self.y])
                if seq == "B" and self.y < len(self.lines) - 1:
                    self.y += 1
                    print("\033[B", end="", flush=True)
                    if self.x > len(self.lines[self.y]):
                        self.x = len(self.lines[self.y])
                if seq == "C" and self.x < len(self.lines[self.y]):
                    self.x += 1
                    print("\033[C", end="", flush=True)
                if seq == "D" and self.x > 0:
                    self.x -= 1
                    print("\033[D", end="", flush=True)

                # delete
                """if seq == "~":
                    if self.x == len(self.lines[self.y]) and self.y < len(self.lines) - 1:
                        redraw = [i for i in range(self.y, len(self.lines))]
                        self.lines[self.y] += self.lines[self.y + 1]
                        self.lines.pop(self.y + 1)
                    else:
                        self.lines[self.y] = self.lines[self.y][:self.x] + self.lines[self.y][self.x+1:]
                        redraw.append(self.y)"""
                if seq == "~":
                    if self.x == len(self.lines[self.y]):
                        if self.y < len(self.lines) - 1 and len(self.lines[self.y]) == 0:
                            redraw = [i for i in range(self.y, len(self.lines))]
                            self.lines.pop(self.y)
                    else:
                        self.lines[self.y] = self.lines[self.y][:self.x] + self.lines[self.y][self.x+1:]
                        redraw.append(self.y)

            else:
                if char in string.printable:
                    if len(self.lines[self.y]) < INPUTBOX[2] - INPUTBOX[0]:
                        if self.lines[self.y][self.x:]: redraw.append(self.y)
                        self.lines[self.y] = self.lines[self.y][:self.x] + char + self.lines[self.y][self.x:]
                        self.x += 1
                        print(f"{char}", flush=True, end="")


                    elif char != " " and len(self.lines[self.y]) == INPUTBOX[2] - INPUTBOX[0]:
                        redraw = [i for i in range(self.y, len(self.lines))]
                        self.lines.insert(self.y+1, char)
                        self.y += 1
                        self.x = 0
                        set_position(self.x+1, self.y + 21)
                        print(f"{char}", flush=True, end="")
                        self.x += 1

            """for line in self.lines:
                if line != "":
                    self.status = 2
                    break"""

    def get_status(self):
        return self.status

def getchar():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        char = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return char

def merge(into: dict, ref: dict):
    for key, val in ref.items():
        if not key in into.keys():
            into[key] = val
        elif isinstance(val, dict):
            into[key] = merge(into[key], ref[key])
    return into

def cursor_on(): print('\033[?25h', end="")
def cursor_off(): print('\033[?25l', end="")
def set_position(x, y):
    print(f"\033[{y+1};{x+1}H", flush=True, end="")

def clear_screen(): os.system("clear && printf '\e[3J'")

def show(text, x, y):
    x = x % os.get_terminal_size()[0]
    y = y % os.get_terminal_size()[1]
    set_position(x,y)
    print(text, end="", flush=True)

def clear_block(x1, y1, x2, y2):
    for i in range(y1, y2):
        #show("\xa4"*(x2-x1), x1, i)
        show(" "*(x2-x1), x1, i)

def print_borders(hsplit: int, vsplit: int):
    w, l = os.get_terminal_size()
    show("\u2554" + (w-2)*"\u2550" + "\u2557", 0, 0)
    for i in range(1,hsplit):
        show("\u2551", 0, i)
        show("\u2551", -1, i)
    show("\u2560" + (vsplit-1)*"\u2550" + "\u2566" + (w-vsplit-2)*"\u2550" + "\u2563", 0, hsplit)
    for i in range(hsplit+1, l-1):
        show("\u2551", 0, i)
        show("\u2551", vsplit, i)
        show("\u2551", -1, i)
    show("\u255a" + (vsplit-1)*"\u2550" + "\u2569" + (w-vsplit-2)*"\u2550" + "\u255d", 0, l-1)
    '''
    \u2554\u2550\u2557
    \u2551 \u2551
    \u2560\u2550\u2563
    \u2551 \u2551
    \u255a\u2550\u255d
    '''

def main():
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

    cursor_off()
    clear_screen()
    print_borders(19, os.get_terminal_size()[0]//2)

    if menu is None:
        menu = Menu("Selection Menu")
        menu.construct(tree)
    else:
        menu.deselect_children()
    menu.selected_child = menu.children[0]
    menu.display()
    active = menu
    clear_block(*HELPBOX)
    active.selected_child.display_help()

    while True:
        char = getchar()

        # Ctrl + C
        if ord(char) == 3:
            cursor_on()
            os.system("clear && printf '\033[3J'")
            exit()
        
        # Ctrl + D, for testing
        if ord(char) == 4:
            menu.alter_visibility(adaptive, False)

        # Ctrl + H
        if ord(char) == 8:
            continue

        # Ctrl + S
        if ord(char) == 19:
            with open("save.pkl", "wb") as f:
                pickle.dump(menu, f)

            if menu.status > 0:               
                with open("input.json", "w") as f:
                    json.dump(menu.to_dict(), f, indent=4)
            continue

        # escape
        if ord(char) == 27:
            while True:
                char = getchar()
                if char.isalpha() or char == "~":
                    break
            
            if char == "A":
                active.prev_child()
            if char == "B":
                active.next_child()
            if char == "C":
                active = active.selected_child
                if isinstance(active.children[0], TextField):
                    cursor_on()
                    active.children[0].get_input()
                    menu.display()
                    cursor_off()
                    active = active.parent
                else:
                    active.selected_child = active.vis_children[0]
            if char == "D" and active != menu:
                active.selected_child = None
                active = active.parent
        menu.display()
        clear_block(*HELPBOX)
        active.selected_child.display_help()

if __name__ == "__main__":
    main()
