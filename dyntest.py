import sys, os
import termios, tty
import json, pickle

from constants import Constants

FG = {
    -1: "\033[0m",
    0:  "\033[31m",
    1:  "\033[33m",
    2:  "\033[32m"
}

BG = {
    -1: "\033[0m",
    0:  "\033[41m",
    1:  "\033[43m",
    2:  "\033[42m"
}

DEFAULT = {
    "control": {
        "text": "General",
        "children": {
            "type": {
                "text": "Trajectory type",
                "default": "sh",
                "type": str
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
                "type": list
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
                "type": list
            },
            "qres": {
                "text": "Number of quantum substeps",
                "default": "20",
                "type": int
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
                "text": "Number of states",
                "default": None,
                "type": int
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
            "propmat": {
                "text": "WF coefficients propagator",
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
            "input": {
                "text": "Input file",
                "default": "input.est",
                "type": str
            }
        }
    },
    "hopping": {
        "text": "Surface Hopping",
        "children": {
            "type": {
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

class Menu():
    def __init__(self, text):
        self.text = text
        self.children: list[Selectable | TextField] = []
        self.vis_children: list[Selectable | TextField] = []
        self.selected_child = None
        self.status = 0

    def construct(self, tree: dict):
        for key, val in tree.items():
            default = val.get("default", "")
            text = val.get("text")
            if default: text += f" [{default}]"
            child = Selectable(key, text)
            if children := val.get("children"):
                child.construct(children)
            else:
                child.append_child(TextField(text=default, tp=val.get("type")))
            self.append_child(child)
        self.vis_children = self.children
    
    def display(self):
        os.system("clear")
        self.get_status()

        print("\033[H", end="")
        print(f"\033[4m{self.text}\033[0m")
        print("Navigation: arrow keys; Exit: Ctrl + C; Save: Ctrl + S; Toggle help (not implemented): Ctrl + H.")
        self.alter_visibility(adaptive["on"], adaptive["off"], not self.get_child_by_text("Adaptive stepsize [n]").children[0].lines[0] in Constants.true)
        self.display_children(0)

    def display_children(self, depth: int):
        for i, child in enumerate(self.vis_children):
            set_position(20*depth, i+2)
            if child == self.selected_child:
                print(f"\033[30m{BG[child.status]}{child.text}\033[0m")
                if child.selected_child is not None and not isinstance(child.selected_child, TextField):
                    child.display_children(depth+1)
            else:
                print(f"{FG[child.status]}{child.text}{FG[-1]}")
    
    def alter_visibility(self, on: list, off: list, rev: bool):
        for name in on:
            self.get_child_by_tag(name).visible = not rev
        for name in off:
            self.get_child_by_tag(name).visible = rev

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

        self.status = stat
        return self.status

    def to_dict(self):
        d = {}
        for child in self.vis_children:
            if isinstance(child, TextField):
                out = "\n".join(child.lines)
                if child.input_type == bool: return out in Constants.true
                elif child.input_type == list: return out.split()
                else: return child.input_type(out)
            else:
                temp = child.to_dict()
                d[child.text] = temp
        return d

class Selectable(Menu):
    def __init__(self, tag: str, text: str):
        self.tag = tag
        self.text = text
        self.parent: Selectable | Menu = None
        self.children: list[Selectable | TextField] = []
        self.vis_children: list[Selectable | TextField] = []
        self.selected_child = None
        self.check = False
        self.parent: Selectable | Menu = None
        self.status = 0
        self.visible = True

class TextField():    
    def __init__(self, text: str = "", tp: type = str):
        self.lines = [text or ""]
        self.parent: Selectable = None
        self.y = len(self.lines) - 1
        self.x = len(self.lines[self.y])
        self.check_text = None
        self.visible = True
        self.status = bool(text)
        self.input_type = tp

    def update_input(self):
        set_position(0,20)
        for j in range(20, os.get_terminal_size()[1]-1):
            print("\033[0K")
        set_position(0,20)
        print("Input: ", end="", flush=True)
        for line in self.lines:
            print("\n", flush=True, end="")
            print(line, flush=True, end="")
        set_position(self.x, self.y + 21)
    
    def get_input(self):
        print("\033[?7l", end="", flush=True)
        
        while True:
            self.update_input()

            char = getchar()

            # Ctrl + C
            if ord(char) == 3:
                self.status = check_input(self.lines[0], self.input_type)
                break

            # enter
            if ord(char) == 13:
                print("\033[E", end="", flush=True)
                self.lines.append("")
                self.y += 1
                self.x = 0
                continue

            # backspace
            if ord(char) == 8:
                if self.x == 0:
                    if self.y > 0:
                        self.y -= 1
                        self.x = len(self.lines[self.y])
                        self.lines[self.y] += self.lines[self.y + 1]
                        self.lines.pop(self.y + 1)
                else:
                    self.lines[self.y] = self.lines[self.y][:self.x-1] + self.lines[self.y][self.x:]
                    self.x -= 1
                    print("\033[D", end="", flush=True)
                continue

            # escape sequence
            if ord(char) == 27:
                while True:
                    seq = sys.stdin.read(1)
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
                if seq == "D" and self.  x > 0:
                    self.x -= 1
                    print("\033[D", end="", flush=True)

                # delete
                if seq == "~":
                    if self.x == len(self.lines[self.y]) and self.y < len(self.lines) - 1:
                        self.lines[self.y] += self.lines[self.y + 1]
                        self.lines.pop(self.y + 1)
                    else:
                        self.lines[self.y] = self.lines[self.y][:self.x] + self.lines[self.y][self.x+1:]

            else:
                self.lines[self.y] = self.lines[self.y][:self.x] + char + self.lines[self.y][self.x:]
                self.x += 1
                print(f"{char}", flush=True, end="")

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

def check_input(inp: str, tp: type):
    if tp == bool:
        if inp in Constants.true or Constants.false: return 2
        else: return 0
    try:
        tp(inp)
        if inp: return 2
        else: return 0
    except:
        return 0

def cursor_on(): print('\033[?25h', end="")
def cursor_off(): print('\033[?25l', end="")
def set_position(x, y): print(f"\033[{y+1};{x+1}H", flush=True, end="")

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
    
    if menu is None:
        menu = Menu("Selection Menu")
        menu.construct(tree)
    else:
        menu.deselect_children()
    breakpoint()
    menu.selected_child = menu.children[0]
    menu.display()
    active = menu

    while True:
        char = getchar()

        # Ctrl + C
        if ord(char) == 3:
            cursor_on()
            os.system("clear")
            exit()
        
        # Ctrl + D, for testing
        if ord(char) == 4:
            menu.alter_visibility(adaptive, False)

        # Ctrl + S
        if ord(char) == 19:
            with open("save.pkl", "wb") as f:
                pickle.dump(menu, f)

            if menu.status > 0:               
                with open("input.json", "w") as f:
                    json.dump(menu.to_dict(), f, indent=4)
            
            
        # escape
        if ord(char) == 27:
            while True:
                char = sys.stdin.read(1)
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

if __name__ == "__main__":
    main()