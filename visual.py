import matplotlib.pyplot as plt 
import numpy as np 
import sys, time

filename = sys.argv[1]
inp = sys.argv[2:]

n_plots = 0
i = 0
cols = []
names = []
flags = ["-p"]
while i < len(inp):
    if inp[i] == "-p":
        n_plots += 1
        cols.append([])
        i += 1

        names.append(inp[i])
        i += 1

        while i < len(inp) and inp[i] not in flags:
            cols[-1].append(int(inp[i]))
            i += 1

plt.ion()
fig, axs = plt.subplots(ncols=len(cols), squeeze=False)
lines = [[] for i in range(n_plots)]
for i in range(n_plots):
    for j in range(len(cols[i])):
        temp, = axs[0][i].plot([], [])
        axs[0][i].set_title(names[i])
        lines[i].append(temp)

while True:
    saved = 1
    with open(filename, "r") as file:
        t = []
        data = [[[] for i in cols[j]] for j in range(n_plots)]
        for l, line in enumerate(file):
            if l < saved: continue
            saved += 1
            
            temp = line.strip().split()
            t.append(float(temp[0]))
            for i in range(n_plots):
                for j in range(len(cols[i])):
                    data[i][j].append(float(temp[cols[i][j]]))
    
    for i in range(n_plots):
        for j, col in enumerate(cols[i]):
            lines[i][j].set_xdata(t)
            lines[i][j].set_ydata(data[i][j])
            axs[0][i].relim()
            axs[0][i].autoscale_view()
            fig.canvas.draw() 
            fig.canvas.flush_events() 
    
    time.sleep(1)
