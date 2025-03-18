
import curses
from curses.textpad import Textbox

class Element:
    def __init__(self, name: str, **kwargs):
        self._setup(name, **kwargs)

    @property
    def status(self):
        return self._status

    def _setup(self, name, root = False, loc = (0, 0), dim = (0, 0)):
        self._status = -1
        self.name = name
        self.root = root
        self.window = curses.newwin(*dim, *loc)
        self.window.keypad(1)

    def display(self):
        self.window.clear()

        while True:
            self.window.refresh()
            self.window.border()
            self.window.addstr(1, 2, Path.path, curses.A_UNDERLINE)
            curses.doupdate()

            self._draw()
            key = self.window.getch()
            if not self._process_input(key):
                break

    def _set_invisible(self):
        self.window.erase()
        self.window.refresh()

    def _draw(self):
        pass

    def _process_input(self):
        pass

class Menu(Element):
    def __init__(self, name, children = None, **kwargs):
        super().__init__(name, **kwargs)
        self.position = 0
        self.children: list[Menu] = children

    def __call__(self):
        self.display()

    @property
    def status(self):
        return min([child.status for child in self.children])

    def navigate(self, n):
        self.position += n
        self.position %= len(self.children)

    def _get_mode(self, index):
        if index == self.position:
            return curses.A_REVERSE
        else:
            return curses.A_NORMAL

    def _print_name(self, index, child: Element, mode):
        msg = f"{child.name}"
        self.window.addstr(2 + index, 2, msg, mode)

    def _draw(self):
        for index, child in enumerate(self.children):
            # mode = self._get_mode(index)
            mode = self._get_mode(index) | curses.color_pair(child.status)
            self._print_name(index, child, mode)

    def _process_input(self, key):
        if key == curses.KEY_RIGHT:
            child = self.children[self.position]
            Path.add(child.name)
            child()
            Path.remove()

        elif key == curses.KEY_LEFT:
            if not self.root:
                return 0

        elif key == curses.KEY_UP:
            self.navigate(-1)

        elif key == curses.KEY_DOWN:
            self.navigate(1)

        elif key == ord("q"):
            exit()

        return 1

class Check(Menu):
    def __init__(self, name, options: list[str], multiple = False, selected = None, **kwargs):
        self._setup(name, **kwargs)
        self.position = 0
        self.children = options
        if selected is None:
            selected = [0]
        self.selected = selected
        self.multiple = multiple
        self._status = 2

    @property
    def status(self):
        return self._status

    def __call__(self):
        self._status = 3
        super().__call__()

    def select(self, index):
        if self.multiple:
            if index in self.selected:
                self.selected.remove(index)
            else:
                self.selected.append(index)
        else:
            self.selected[0] = index

    def _print_name(self, index, child, mode):
        msg = f"  {child.name}"
        self.window.addstr(2 + index, 2, msg, mode)

        if index in self.selected:
            self.window.addstr(2 + index, 2, "> ")
        else:
            self.window.addstr(2 + index, 2, "  ")

    def _process_input(self, key):
        if key == curses.KEY_RIGHT:
            self.select(self.position)

        elif key == curses.KEY_LEFT:
            self._set_invisible()
            return 0

        elif key == curses.KEY_UP:
            self.navigate(-1)

        elif key == curses.KEY_DOWN:
            self.navigate(1)

        return 1

class Item(Element):
    def __init__(self, name, action = lambda: None, **kwargs):
        self._setup(name, **kwargs)
        self.action = action
        self._status = 0

    def __call__(self):
        self.action()

class Text(Element):
    def __init__(self, name, text = "", **kwargs):
        self._setup(name, **kwargs)
        self._status = 2
        if text is None:
            self._status = 1
            text = ""
        self.text = text
        h, w = self.window.getmaxyx()
        h -= 3
        w -= 4
        self.boxwin = self.window.derwin(h, w, 2, 2)
        self.box = Textbox(self.boxwin, insert_mode=True)


    def __call__(self):
        self.window.border()
        self.window.addstr(1, 2, Path.path, curses.A_UNDERLINE)
        self.window.refresh()

        curses.curs_set(1)
        self.boxwin.move(0, 0)
        self.box.edit()
        self.text = self.box.gather()
        curses.curs_set(0)

        self._set_invisible()

        # TODO: validation
        if self.text:
            self._status = 3

class Help(Element):
    def display(self):
        self.window.addstr(1, 2, "HELP", curses.A_UNDERLINE)
        self.window.addstr(2, 2, "Navigate using arrow keys")
        self.window.addstr(3, 2, " - move up and down using the corresponding arrow keys")
        self.window.addstr(4, 2, " - traverse the hierarchy with left/right arrow keys")
        self.window.addstr(5, 2, "To (de)select one of multiple options, use right arrow key")
        self.window.addstr(6, 2, "To exit text editor, use Ctrl + G")
        self.window.addstr(7, 2, "The text editor uses other basic Emacs key bindings")
        self.window.addstr(8, 2, "Press S to save")
        self.window.addstr(9, 2, "Press Q to quit")
        self.window.addstr(10, 2, "Press R to switch between optional / mandatory mode [TODO]")

        self.window.border()
        self.window.refresh()

class Path:
    path = "MAIN"

    @classmethod
    def add(cls, val: str):
        cls.path += f" > {val.upper()}"

    @classmethod
    def remove(cls):
        cls.path = cls.path.rsplit(" > ", maxsplit=1)[0]

class App:
    DEFAULT = {
        "sampling": {
            "distr": ("wigner", "husimi", "ho"),
            "input": None,
            "samples": None,
            "emin": None,
            "emax": None,
            "from": None,
            "to": None
        },

        "dynamics": {
            "name": "",
            "method": ("fssh", "msh", "ehr", "csdm"),
            "pop_est": ("wigner", "semiclassical", "spinmap"),
            "prob": ("tdc", "none", "prop", "gf"),
            "seed": "",
            "decoherence": ("none", "edc"),
            "initstate": "",
            "backup": "true",
            "tmax": None,
            "dt": None,
            "timestep": "const",
            "enthresh": "",
            "max_depth": ""
        },

        "nuclear": {
            "input": "geom.xyz",
            "nuc_upd": "vv",
            "com": "true"
        },

        "quantum": {
            "input": "",
            "tdc_upd": ("npi", "none", "nacme", "hst", "hst3", "npisharc", "npimeek", "ktdce", "ktdcg"),
            "coeff_upd": ("tdc", "none", "tdc", "ld"),
            "n_substeps": "50",
        },

        "electronic": {
            "program": ("molpro", "molcas"),
            "path": "",
            "method": None,
            "states": None,
            "options": {
                "basis": None,
                "closed": None,
                "active": None,
                "sa": None,
                "nel": None,
                "df": "false",
                "dfbasis": "avdz"
            }
        },

        "output": {
            "file": "out",
            "log": "true",
            "verbosity [TODO]": (1,2,3),
            "dat": "true",
            "record": "",
            "h5": "true",
            "xyz": "true",
            "dist": "false"
        }
    }

    def __init__(self, stdscreen: curses.window):
        self.screen = stdscreen

        curses.curs_set(0)
        curses.init_pair(1, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_GREEN, curses.COLOR_BLACK)

        help = Help("help", **self.UR)
        help.display()

        menu = self.construct("menu", self.DEFAULT, root = True)
        menu.display()

    @property
    def height(self):
        return self.screen.getmaxyx()[0]

    @property
    def width(self):
        return self.screen.getmaxyx()[1]

    @property
    def UL(self):
        return {
            "loc": (0, 0),
            "dim": (self.height//2, self.width//2)
        }

    @property
    def LL(self):
        return {
            "loc": (self.height//2, 0),
            "dim": (self.height//2, self.width//2)
        }

    @property
    def UR(self):
        return {
            "loc": (0, self.width//2),
            "dim": (self.height//2, self.width//2)
        }

    @property
    def LR(self):
        return {
            "loc": (self.height//2, self.width//2),
            "dim": (self.height//2, self.width//2)
        }



    def construct(self, name, inp, root = False):
        if type(inp) == dict:
            elem = Menu(name, [self.construct(key, val) for key, val in inp.items()], root = root, **self.UL)
        elif type(inp) == list:
            elem = Check(name, [Item(x) for x in inp], multiple=True, **self.LL)
        elif type(inp) == tuple:
            elem = Check(name, [Item(x) for x in inp], **self.LL)
        elif inp is None or type(inp) == str:
            elem = Text(name, inp, **self.LL)
        return elem

if __name__ == "__main__":
    curses.wrapper(App)
