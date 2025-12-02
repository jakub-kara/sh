from textual.app import App
from textual.widgets import Button, Static, Header, Footer, Label, RadioSet, RadioButton, Checkbox, TextArea, Input, LoadingIndicator
from textual.containers import Container
from textual.screen import ModalScreen
from textual.message import Message
from textual.reactive import reactive
from textual import work
from textual.worker import Worker, WorkerState

import json
import importlib
from classes.config import Config
from classes.meta import get_children, get_parent

class Result(Config):
    cfg = {}
    res = ...
    viewed = False

    @property
    def state(self):
        if not self.is_valid:
            return 0
        elif self.viewed:
            return 2
        else:
            return 1

    @property
    def is_valid(self):
        tmp, _ = self.valid.check_convert(self.res)
        return tmp

    @property
    def is_fixed(self):
        return self.fix is not ...

    @property
    def has_default(self):
        return self.dfl is not ...

    @staticmethod
    def cfg_to_res(cfg: Config) -> "Result":
        dic = Result.cfg
        if cfg.path is not None:
            for k in cfg.path.split("."):
                if k in dic:
                    dic = dic[k]
            return dic[cfg.key]

    def set_res(self):
        if self.has_default:
            self.res = self.dfl
        if self.is_fixed:
            self.res = self.fix

    @staticmethod
    def find_root(kls):
        for k in kls.mro():
            for key, res in Result.dic.items():
                if key.startswith(k.__qualname__) and res.root:
                    return res
        return None

    @staticmethod
    def _is_root(kls):
        root = Result.find_root(kls)
        if root is None:
            return False
        return root.kls is kls

    @staticmethod
    def _is_sel(kls: type):
        issel = not (any(c in Result.sel for c in get_children(kls)) or kls in Result.sel) or Result._is_root(kls)
        for c in [kls] + list(get_children(kls)):
            sel = Result.sel.get(c)
            if sel is not None:
                temp = Result.cfg_to_res(sel)
                if temp is not None:
                    issel = issel or (sel.opt.get(temp.res) is c)
        return issel

    @staticmethod
    def _is_cmp(kls: type):
        iscmp = False
        for p in kls.mro():
            comp = Result.comp.get(p)
            if comp is p:
                return True
            elif comp is not None:
                iscmp = iscmp or Result.is_vis(comp)
        return iscmp

    @staticmethod
    def is_vis(kls: type):
        return Result._is_sel(kls) and Result._is_cmp(kls)

        # out = not (any(c in Result.sel for c in get_children(kls)) or kls in Result.sel)
        # for c in list(get_children(kls)) + [kls]:
        #     sel = Result.sel.get(c, None)
        #     if sel is None:
        #         continue
        #     temp = Result.cfg_to_res(sel)
        #     if temp is None:
        #         continue
        #     out = out or sel.opt.get(temp.res, ...) is c

        # cmp = not any(p in Result.comp for p in kls.mro())
        # for p in kls.mro():
        #     if p in Result.comp:
        #         temp = Result.comp[p]
        #         if temp is p:
        #             cmp = True
        #         else:
        #             vis = Result.is_vis(temp)
        #             cmp = cmp or vis
        # return out and cmp

    @property
    def vis(self):
        return Result.is_vis(self.kls)

    def to_str(self):
        if self.res is ... or self.res is None:
            return ""
        elif isinstance(self.res, list):
            return "\n".join([str(r) for r in self.res])
        else:
            return str(self.res)

    @classmethod
    def load_cfg(cls):
        with open("vis.txt", "w") as f: pass

        cfg = {}
        for key, val in Config.dic.items():
            path = val.path
            if path is None:
                continue

            if path not in cfg:
                cfg[path] = {}
            val.__class__ = Result
            val.set_res()
            # key: [config, result]
            cfg[path][key.split(".")[-1]] = val
        cls.cfg = cfg

def save_res(dic: dict):
    def _walk(x: dict):
        temp = {}
        for k, v in x.items():
            if isinstance(v, dict):
                temp[k] = _walk(v)
            elif v.vis and v.state != 1:
                temp[k] = v.res if v.res is not ... else None
        return temp
    return _walk(dic)

class PopupScreen(ModalScreen):
    pass

class PopupBody(Container):
    pass

class InputScreen(PopupScreen):
    class Enter(Message):
        def __init__(self, obj):
            super().__init__()
            self.obj = obj

    class Exit(Message):
        pass

    def __init__(self, obj: Config, **kwargs):
        super().__init__(**kwargs)
        self._box = InputBox(obj)

    def compose(self):
        yield self._box

class InputBox(PopupBody):
    def __init__(self, obj, **kwargs):
        super().__init__(**kwargs)
        self._obj: Result = obj
        self._prompt = None
        self.out = None

    def compose(self):
        yield Label(self._obj.key, classes="title")
        yield Label("Some description", classes="InputDesc")
        # works for both dict and list
        if len(self._obj.opt) > 0:
            self._prompt = RadioSet()
            with self._prompt:
                for opt in self._obj.opt:
                    opt = str(opt)
                    temp = RadioButton(opt, id=opt)
                    if opt == self._obj.to_str():
                        temp.value = True
                    yield temp
        elif type(self._obj.valid) is type(Config.LIST):
            self._prompt = TextArea(self._obj.to_str())
            yield self._prompt
        else:
            self._prompt = Input(self._obj.to_str())
            yield self._prompt
        yield Label("Option description", classes="OptionDesc")

    def key_escape(self):
        valid = True
        if isinstance(self._prompt, RadioSet):
            out = None
            if self._prompt.pressed_button:
                out = self._prompt.pressed_button.id
        elif isinstance(self._prompt, TextArea):
            out = self._prompt.text
            valid, _ = self._obj.valid.check_convert(out)
        elif isinstance(self._prompt, Input):
            out = self._prompt.value
            valid, _ = self._obj.valid.check_convert(out)

        # if valid:
        self._obj.res = out
        self._obj.viewed = True
        self.post_message(InputScreen.Exit())

class Body(Static):
    cfg = {}
    cfg_path = reactive(list, recompose=True)
    BINDINGS = [("h", "toggle", "Hide Completed")]
    _states = ["invalid", "default", "valid"]
    _hide_completed = reactive(False, recompose=True)
    _active_btn = None

    @property
    def state(self):
        return self._get_state(Result.cfg)

    def action_toggle(self):
        self._hide_completed = not self._hide_completed

    def _get_state(self, dic: dict):
        state = 2
        if isinstance(dic, Result):
            return dic.state

        for key, res in dic.items():
            temp = 2
            if isinstance(res, dict):
                temp = self._get_state(res)
            elif res.vis:
                temp = res.state
            state = min(state, temp)
        return state

    @classmethod
    def set_cfg(cls):
        if Result.cfg == {}:
            Result.load_cfg()

    def compose(self):
        dic = Result.cfg
        for p in self.cfg_path:
            dic = dic[p]

        if isinstance(dic, dict):
            num = 0
            foc = True
            for key, res in dic.items():
                if isinstance(res, Result):
                    if not res.vis:
                        continue

                state = self._get_state(res)
                if state in (1,2) and self._hide_completed:
                    continue
                classes = self._states[state]

                temp = Button(key, id=key, classes=classes)

                if foc:
                    temp.focus()
                    foc = False
                num += 1
                yield temp

            if num == 0:
                self.key_left()
        else:
            self.post_message(InputScreen.Enter(dic))


    def key_left(self):
        if self.cfg_path != []:
            self.cfg_path.pop()
            self.mutate_reactive(Body.cfg_path)

    def key_right(self):
        for child in self.children:
            if child.has_focus:
                child.press()
                return

    def on_button_pressed(self, event: Button.Pressed):
        self.cfg_path.append(event.button.id)
        self.mutate_reactive(Body.cfg_path)

class LoadingScreen(PopupScreen):
    class Exit(Message):
        pass

    def compose(self):
        yield LoadingBody(
            Label("Loading...", classes="title"),
            LoadingIndicator()
        )

    def on_mount(self):
        self._load_config("imports")
        # self._load_config("config_test")

    @work(thread=True, name="imports")
    def _load_config(self, module_name: str):
        importlib.import_module(module_name)

    def on_worker_state_changed(self, event: Worker.StateChanged):
        if event.worker.name == "imports":
            if event.state is WorkerState.SUCCESS:
                self.app.pop_screen()
                self.post_message(LoadingScreen.Exit())
            elif event.state is WorkerState.ERROR:
                self.mount(Label(f"Failed to load: {event.worker.error!r}"))

class LoadingBody(PopupBody):
    pass

class SavedScreen(PopupScreen):
    def __init__(self, state, **kwargs):
        super().__init__(**kwargs)
        self._state = state

    def compose(self):
        content = [Label("Saved")]
        if self._state == 0:
            content.append(Label("Warning: Some values were not specified, setting to null."))
        elif self._state == 1:
            content.append(Label("Warning: Not all values were inspected, setting to default."))
        content.append(Label("Press any key to continue"))
        yield PopupBody(
            *content
        )

    def on_key(self, event):
        self.app.pop_screen()
        event.stop()

class GUI(App):
    CSS_PATH = "styles.tcss"
    BINDINGS = [
        # ("up", "key_up", ""),
        ("q", "key_q", "Exit")
    ]

    def compose(self):
        yield Header()
        self.body = Body()
        yield self.body
        yield Footer()

    def on_mount(self):
        self.push_screen(LoadingScreen())

    def on_loading_screen_exit(self, msg: LoadingScreen.Exit):
        self.body.set_cfg()
        self.body.mutate_reactive(Body.cfg_path)

    def on_input_screen_enter(self, msg: InputScreen.Enter):
        self.push_screen(InputScreen(msg.obj))

    def on_input_screen_exit(self, msg: InputScreen.Exit):
        self.pop_screen()
        self.body.key_left()

    def key_up(self):
        self.action_focus_previous()

    def key_down(self):
        self.action_focus_next()

    def key_q(self):
        self.exit()

    def key_s(self):
        self.push_screen(SavedScreen(self.body.state))
        dic = save_res(Result.cfg)
        with open("input.json", "w") as f:
            json.dump(dic, f, indent=4)

if __name__ == "__main__":
    app = GUI()
    app.run()
    breakpoint()