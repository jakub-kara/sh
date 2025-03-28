if _name_ == "_main_":
    with open("geom", "r") as f:
        geom = f.readlines()
    with open("veloc", "r") as f:
        veloc = f.readlines()
    with open("geom.xyz", "w") as f:
        f.write(f"{len(geom)}\n\n")
        for i in range(len(geom)):
            data = geom[i].strip().split()
            f.write(f"{data[0]} {data[2]} {data[3]} {data[4]} {' '.join(veloc[i].strip().split())}\n")