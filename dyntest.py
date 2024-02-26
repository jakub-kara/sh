import sys, os
import termios, tty

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

tree = {
    "General": {
        "Trajectory Type [sh]": "sh",
        "Path to ensemble [.]": ".",
        "Name of molecule [x]": "x",
        "Quantities to record [pes]": "pes",
        "Input units [au]": "au",
        "Total time": None,
        "Adaptive stepsize [n]": "n",
        "Stepsize": None,
    },
    "Nuclear Settings": {
        "Input format": None,
        "Nuclear propagator": None
    }
}

adaptive = {
    "Stepsize function [tanh]": "tanh",
    "Stepsize variable [nac**2]": "nac**2",
    "Max stepsize": None,
    "Min stepsize": None,
    "Stepsize parameters": None,
}

class Menu():
    def __init__(self, text):
        self.text = text
        self.children: list[Selectable | TextField] = []
        self.selected_child = 0
        self.status = 0

    def construct(self, tree: dict):
        for key, val in tree.items():
            self.append_child(Selectable(key))
            if isinstance(val, dict):
                for key2, val2 in val.items():
                    if val2 is None:
                        self.children[-1].append_child(Selectable(key2).append_child(TextField()))
                    else:
                        self.children[-1].append_child(Selectable(key2).append_child(TextField(text=val2)))
                        self.children[-1].children[-1].status = 1
            else:
                self.children[-1].append_child(TextField())
    
    def display(self):
        os.system("clear")
        self.get_status()

        print("\033[H", end="")
        print(f"\033[4m{self.text}\033[0m")
        print("Navigation: arrow keys; Exit: Ctrl + C; Save: Ctrl + S; Toggle help (not implemented): Ctrl + H.")
        self.display_children(0)
        
    def append_child(self, child):
        self.children.append(child)
        child.parent = self
        return self
    
    def get_child_by_text(self, text: str):
        for child in self.children:
            if text in child.text:
                return child
    
    def remove_child(self, child):
        for i, other in enumerate(self.children):
            if other == child:
                self.children.pop(i)
                break
        del child

    def display_children(self, depth: int):
        for i, child in enumerate(self.children):
            set_position(20*depth, i+2)
            if self.selected_child == i:
                print(f"\033[30m{BG[child.status]}{child.text}\033[0m")
                if child.selected_child != -1 and not isinstance(child.children[child.selected_child], TextField):
                    child.display_children(depth+1)
            else:
                print(f"{FG[child.status]}{child.text}{FG[-1]}")

    def get_status(self):
        stat = 2
        for child in self.children:
            stat = min(child.get_status(), stat)

        self.status = stat
        return self.status

    def save(self):
        with open("save.dat", "w") as f:
            pass

class Selectable(Menu):
    def __init__(self, text):
        self.text = text
        self.parent: Selectable | Menu = None
        self.children: list[Selectable | TextField] = []
        self.selected_child = -1
        self.check = False
        self.parent: Selectable | Menu = None
        self.status = 0

class TextField():    
    def __init__(self, text = ""):
        self.lines = [text]
        self.parent: Selectable = None
        self.y = len(self.lines) - 1
        self.x = len(self.lines[self.y])
        self.status = bool(text)

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

            for line in self.lines:
                if line != "":
                    self.status = 2
                    break

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

def cursor_on(): print('\033[?25h', end="")
def cursor_off(): print('\033[?25l', end="")
def set_position(x, y): print(f"\033[{y+1};{x+1}H", flush=True, end="")

def main():
    cursor_off()
    
    menu = Menu("Selection Menu")
    menu.construct(tree)
    menu.display()
    active = menu

    while True:
        char = getchar()

        # Ctrl + C
        if ord(char) == 3:
            cursor_on()
            os.system("clear")
            menu.save()
            exit()

        # Ctrl + S
        if ord(char) == 19:
            menu.save()
        
        # escape
        if ord(char) == 27:
            while True:
                char = sys.stdin.read(1)
                if char.isalpha() or char == "~":
                    break
            
            if char == "A":
                active.selected_child -= 1
                active.selected_child %= len(active.children)
                menu.display()
            if char == "B":
                active.selected_child += 1
                active.selected_child %= len(active.children)
                menu.display()
            if char == "C":
                active = active.children[active.selected_child]
                if isinstance(active.children[0], TextField):
                    cursor_on()
                    active.children[0].get_input()
                    cursor_off()
                    active = active.parent
                    menu.display()
                else:
                    active.selected_child = 0
                    menu.display()
            if char == "D" and active != menu:
                active.selected_child = -1
                active = active.parent
                menu.display()
        

if __name__ == "__main__":
    main()