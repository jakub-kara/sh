from classes.config import Config


@Config("root", path=Config.DYNAMICS)
@Config("x", default=0, valid=Config.INT, path=Config.DYNAMICS)
@Config.Init(Config.Init.THIS)
class Root:
    pass

@Config("asub", path=Config.DYNAMICS)
class Asub:
    pass


@Config.Init(Asub)
class A(Root):
    pass

@Config.Option("a1", "root")
class A1(A):
    pass


@Config.Option("b", "root")
@Config("b_setting", path=Config.DYNAMICS)
class B(Root):
    pass


@Config.Option("opt1", "asub")
class Opt1(Asub):
    pass

@Config.Option("opt2", "asub")
@Config("param", path=Config.DYNAMICS)
class Opt2(Asub):
    pass