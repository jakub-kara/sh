import curses
from curses.textpad import Textbox
import json
from classes.constants import convert

class Element:
    def __init__(self, name: str, **kwargs):
        self._setup(name, **kwargs)
        self.visible = True

    @property
    def status(self):
        return self._status

    def _setup(self, name, desc = "", root = False, loc = (0, 0), dim = (0, 0)):
        self._status = -1
        self.name = name
        self.root = root
        desc = f"some description for {self.name}"
        self.desc = desc
        self.window = curses.newwin(*dim, *loc)
        self.window.keypad(1)

    def display(self):
        self.window.clear()
        desc.show_content(self)
        while True:
            self.window.refresh()
            self.window.border()
            self.window.addstr(1, 2, Path.path, curses.A_UNDERLINE)
            curses.doupdate()

            self._draw()
            key = self.window.getch()
            if not self._process_input(key):
                break

    def erase(self):
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
        if children is None:
            children = []
        self.children: list[Menu] = children

    def __call__(self):
        self.display()

    @property
    def status(self):
        return min([child.status for child in self.children])

    @property
    def vis_children(self):
        return [i for i in self.children if i.visible and (i.status < 2 or Observer.optional)]

    def navigate(self, n):
        self.position += n
        self.position %= len(self.vis_children)

    def _get_mode(self, index):
        if index == self.position:
            return curses.A_REVERSE
        else:
            return curses.A_NORMAL

    def _print_name(self, index, child: Element, mode):
        msg = f"{child.name}"
        self.window.addstr(2 + index, 2, msg, mode)

    def _draw(self):
        for index, child in enumerate(self.vis_children):
            mode = self._get_mode(index) | curses.color_pair(child.status)
            self._print_name(index, child, mode)

    def _erase_children(self, children):
            for index, child in enumerate(children):
                self.window.addstr(2 + index, 2, " "*len(child.name))

    def save(self):
        out = {}
        for child in self.vis_children:
            # if child.status == 2:
            #     continue
            temp = child.save()
            out[child.name] = temp
        return out

    def _process_input(self, key):
        if key == curses.KEY_RIGHT:
            child = self.vis_children[self.position]
            temp = self.vis_children.copy()
            Path.add(child.name)
            child()
            self._erase_children(temp)
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

        elif key == ord("r"):
            self._erase_children(self.vis_children)
            Observer.optional = not Observer.optional
            self.navigate(-1)

        elif key == ord("s"):
            out = self.save()
            with open("input.json", "w") as f:
                json.dump(out, f, indent=4)
        return 1

class Radio(Menu):
    def __init__(self, name, options: list[str], selected = 0, **kwargs):
        Element.__init__(self, name, **kwargs)
        self.position = 0
        self.children = options
        self.selected = selected
        self._status = 2

    @property
    def status(self):
        return self._status

    @property
    def val(self):
        return self.vis_children[self.selected].val

    def __call__(self):
        self._status = 3
        super().__call__()

    def _print_name(self, index, child, mode):
        msg = f"  {child.name}"
        self.window.addstr(2 + index, 2, msg, mode)

        if index == self.selected:
            self.window.addstr(2 + index, 2, "> ")
        else:
            self.window.addstr(2 + index, 2, "  ")

    def _process_input(self, key):
        if key == curses.KEY_RIGHT:
            self.selected = self.position
            Observer.alert(self)

        elif key == curses.KEY_LEFT:
            self.erase()
            return 0

        elif key == curses.KEY_UP:
            self.navigate(-1)

        elif key == curses.KEY_DOWN:
            self.navigate(1)

        return 1

    def save(self):
        return self.val

class Item(Element):
    def __init__(self, name, val = None, action = lambda: None, **kwargs):
        super().__init__(name, **kwargs)
        self.action = action
        if val is None:
            val = name
        self.val = val
        self._status = 0

    def save(self):
        return self.val

    def __call__(self):
        self.action()

class Text(Element):
    def __init__(self, name, text = "", valid = "", multiple = False, empty_ok = False, **kwargs):
        super().__init__(name, **kwargs)
        self._status = 2
        self.text = str(text)
        self.val = None
        self._emptyok = empty_ok
        h, w = self.window.getmaxyx()
        h -= 3
        w -= 4
        self.boxwin = self.window.derwin(h, w, 2, 2)
        self.box = Textbox(self.boxwin)

        conv = {
            "": lambda x: x,
            "int": Text.to_int,
            "float": Text.to_float,
            "pos": Text.check_pos,
            "unit": Text.check_unit,
        }
        self._convs = [conv[i] for i in valid.split()]
        self._multi = multiple
        self.convert()
        if self._status > 2:
            self._status = 2

    def save(self):
        return self.val

    def __call__(self):
        self.window.border()
        self.window.addstr(1, 2, Path.path, curses.A_UNDERLINE)
        self.window.refresh()

        curses.curs_set(1)
        self.boxwin.move(0, 0)
        self.boxwin.addstr(self.text)
        self.box.edit()
        self.text = self.box.gather()
        self.convert()
        curses.curs_set(0)

        Observer.alert(self)
        self.erase()

    def convert(self):
        self.val = [i.strip() for i in self.text.replace("\n", ",").split(",")]
        self.val = [i for i in self.val if i != ""]
        for conv in self._convs:
            self.val = [conv(i) for i in self.val]

        self._status = 1 + 2 * all([i is not None for i in self.val]) * bool(self.val)
        if not self._multi:
            self._status = 1
            if len(self.val) == 1:
                self.val = self.val[0]
                self._status = 3
            else:
                self.val = None
        if self._emptyok:
            self._status = 3

    @staticmethod
    def to_int(x):
        try: return int(x)
        except: return None

    @staticmethod
    def to_float(x):
        try: return float(x)
        except: return None

    @staticmethod
    def check_pos(x):
        if x is None: return None
        if x >= 0: return x
        else: return None

    @staticmethod
    def check_unit(x):
        try:
            return convert(x, "au")
        except:
            return None

class Help(Element):
    def display(self):
        self.window.addstr(1, 2, "HELP", curses.A_UNDERLINE)
        self.window.addstr(2, 2, "Navigate using arrow keys")
        self.window.addstr(3, 2, " - move up and down using the corresponding arrow keys")
        self.window.addstr(4, 2, " - traverse the hierarchy with left/right arrow keys")
        self.window.addstr(5, 2, "To select an option, use right arrow key")
        self.window.addstr(6, 2, "To exit text editor, use Ctrl + G")
        self.window.addstr(7, 2, " - the text editor uses other standard Emacs key bindings")
        self.window.addstr(8, 2, "Press S to save")
        self.window.addstr(9, 2, "Press Q to quit")
        self.window.addstr(10, 2, "Press R to switch between optional / mandatory mode")

        self.window.border()
        self.window.refresh()

class Description(Element):
    def display(self):
        self.window.addstr(1, 2, "DESCRIPTION", curses.A_UNDERLINE)
        self.window.border()
        self.window.refresh()

    def show_content(self, elem: Element):
        self.window.clear()
        self.window.addstr(1, 2, "DESCRIPTION", curses.A_UNDERLINE)
        self.window.border()
        for i, line in enumerate(elem.desc.split("\n")):
            self.window.addstr(2+i, 2, line)
        self.window.refresh()

class Path:
    path = "MAIN"

    @classmethod
    def add(cls, val: str):
        cls.path += f" > {val.upper()}"

    @classmethod
    def remove(cls):
        cls.path = cls.path.rsplit(" > ", maxsplit=1)[0]

class Observer:
    _subs: dict[Element, list[tuple]] = {}
    optional = True

    @classmethod
    def subscribe(cls, sub: Element, src: Element, *conds):
        if src in cls._subs.keys():
            cls._subs[src].append((sub, conds))
        else:
            cls._subs[src] = [(sub, conds)]

    @classmethod
    def alert(cls, src):
        if src not in cls._subs.keys():
            return
        for sub, conds in cls._subs[src]:
            sub.visible = src.val in conds and src.visible
            Observer.alert(sub)

class App:
    def __init__(self, stdscreen: curses.window):
        self.screen = stdscreen

        curses.curs_set(0)
        curses.init_pair(1, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_GREEN, curses.COLOR_BLACK)

        help = Help("help", **self.UR)
        help.display()

        global desc
        desc = Description("description", **self.LR)
        desc.display()

        menu = self.construct()
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

    def construct(self):
        main = Menu("menu", root = True, **self.UL)
        sampling = Menu("sampling", **self.UL)
        sampling.children.extend([
            Radio("distr", [
                Item("wigner"),
                Item("husimi"),
                Item("ho")], **self.LL),
            Text("input", **self.LL),
            Text("samples", "", "int pos", **self.LL),
            Text("emin", "", "float pos", **self.LL),
            Text("emax", "", "float pos", **self.LL),
            Text("from", "", "int pos", True, **self.LL),
            Text("to", "", "int", True, **self.LL)
        ])

        dynamics = Menu("dynamics", **self.UL)
        dynamics.children.extend([
            Text("name", "", **self.LL),
            method:=Radio("method", [
                Item("fssh"),
                Item("mash"),
                Item("ehr"),
                Item("csdm"),
                Item("lscivr")], **self.LL),
            popest:=Radio("pop_est", [
                Item("wigner"),
                Item("semiclassical"),
                Item("spinmap")], **self.LL),
            prob:=Radio("prob", [
                Item("none"),
                Item("tdc"),
                Item("prop"),
                Item("gf")], selected=1, **self.LL),
            seed:=Text("seed", "", **self.LL),
            deco:=Radio("decoherence", [
                Item("none"),
                Item("edc")], selected=1, **self.LL),
            Text("initstate", "", "int pos", **self.LL),
            Radio("backup", [
                Item("True", True),
                Item("False", False)], **self.LL),
            Text("tmax", "", "unit float pos", **self.LL),
            Text("dt", "", "unit float pos", **self.LL),
            timestep:=Radio("timestep", [
                Item("const"),
                Item("half")], **self.LL),
            enthresh:=Text("enthresh", "1e10", **self.LL),
            depth:=Text("max_depth", 10, **self.LL)
        ])
        Observer.subscribe(popest, method, "lscivr")
        Observer.subscribe(prob, method, "fssh", "csdm")
        Observer.subscribe(seed, method, "fssh", "csdm")
        Observer.subscribe(deco, method, "fssh")
        Observer.alert(method)

        Observer.subscribe(enthresh, timestep, "half")
        Observer.subscribe(depth, timestep, "half")
        Observer.alert(timestep)

        nuclear = Menu("nuclear", **self.UL)
        nuclear.children.extend([
            Text("input", "geom.xyz", **self.LL),
            Radio("nuc_upd", [
                Item("vv"),
                Item("syam4"),
                Item("rkn4"),
                Item("y4")], **self.LL),
            Radio("com", [
                Item("True", True),
                Item("False", False)], **self.LL)
        ])

        quantum = Menu("quantum", **self.UL)
        quantum.children.extend([
            Text("input", empty_ok=True, **self.LL),
            Radio("tdc_upd", [
                Item("none"),
                Item("hst"),
                Item("hst3"),
                Item("npi"),
                Item("npisharc"),
                Item("npimeek"),
                Item("ktdce"),
                Item("ktdcg"),
                Item("nacme")], selected=3, **self.LL),
            Radio("coeff_upd", [
                Item("none"),
                Item("tdc"),
                Item("ld")], selected=1, **self.LL),
            Text("n_substeps", 50, "int pos", **self.LL)
        ])

        options = Menu("options", **self.UL)
        options.children.extend([
            Text("basis", **self.LL),
            closed:=Text("closed", "", "int pos", **self.LL),
            active:=Text("active", "", "int pos", **self.LL),
            sa:=Text("sa", "", "int pos", **self.LL),
            nel:=Text("nel", "", "int pos", **self.LL),
            df:=Radio("df", [
                Item("True", True),
                Item("False", False)], **self.LL),
            dfbasis:=Text("dfbasis", "avdz", **self.LL)
        ])

        electronic = Menu("electronic", **self.UL)
        electronic.children.extend([
            program:=Radio("program", [
                Item("molpro"),
                Item("molcas")], **self.LL),
            Text("path", "", **self.LL),
            method:=Radio("method", [
                cas:=Item("cas"),
                pt2:=Item("pt2")], **self.LL),
            Text("states", None, "int pos", True, **self.LL),
            options
        ])
        Observer.subscribe(pt2, program, "molcas")
        Observer.alert(program)
        [Observer.subscribe(x, method, "cas", "pt2") for x in [closed, active, sa, nel]]
        Observer.alert(method)
        Observer.subscribe(dfbasis, df, True)
        Observer.alert(df)

        output = Menu("output", **self.UL)
        output.children.extend([
            Text("file", "out", **self.LL),
            Radio("log", [
                Item("True", True),
                Item("False", False)], **self.LL),
            Radio("verbosity [TODO]", [
                Item("1"),
                Item("2"),
                Item("3")], **self.LL),
            Radio("dat", [
                Item("True", True),
                Item("False", False)], **self.LL),
            Text("record", "pop, pen, ten", multiple=True, **self.LL),
            Radio("h5", [
                Item("True", True),
                Item("False", False)], **self.LL),
            Radio("xyz", [
                Item("True", True),
                Item("False", False)], **self.LL),
            Radio("dist", [
                Item("True", True),
                Item("False", False)], selected=1, **self.LL),
        ])

        main.children.extend([
            sampling,
            dynamics,
            nuclear,
            quantum,
            electronic,
            output
        ])
        return main

if __name__ == "__main__":
    curses.wrapper(App)