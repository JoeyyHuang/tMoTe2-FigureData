import os
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

fs = 18
nr = 2000

def read_poscar(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    avec = np.zeros((3,3))
    parts = lines[2].split()
    avec[0] = [float(parts[0]), float(parts[1]), float(parts[2])]
    parts = lines[3].split()
    avec[1] = [float(parts[0]), float(parts[1]), float(parts[2])]
    parts = lines[4].split()
    avec[2] = [float(parts[0]), float(parts[1]), float(parts[2])]
    parts = lines[6].split()
    nM = int(parts[0])
    nX = int(parts[1])
    natom = nM + nX
    coords = np.zeros((natom,3))
    for line in range(natom):
        parts = lines[line+8].split()
        coords[line] = [float(parts[0]), float(parts[1]), float(parts[2])]
    return avec, nM, nX, coords

avec, nM, nX, relaxed_coords = read_poscar(os.path.join(".", "relaxed-POSCAR"))
avec, nM, nX, unrelax_coords = read_poscar(os.path.join(".", "unrelax-POSCAR"))

disp_coords = relaxed_coords - unrelax_coords


x2 = unrelax_coords[np.r_[0:nM//2, nM:3*nM//2], 0]
y2 = unrelax_coords[np.r_[0:nM//2, nM:3*nM//2], 1]
xi2 = disp_coords[np.r_[0:nM//2, nM:3*nM//2], 0] * avec[0,0] + disp_coords[np.r_[0:nM//2, nM:3*nM//2], 1] * avec[1,0]
yi2 = disp_coords[np.r_[0:nM//2, nM:3*nM//2], 0] * avec[0,1] + disp_coords[np.r_[0:nM//2, nM:3*nM//2], 1] * avec[1,1]
zi2 = disp_coords[np.r_[0:nM//2, nM:3*nM//2], 2] * avec[2,2]

x1 = unrelax_coords[np.r_[nM//2:nM, 2*nM:5*nM//2], 0]
y1 = unrelax_coords[np.r_[nM//2:nM, 2*nM:5*nM//2], 1]
xi1 = disp_coords[np.r_[nM//2:nM, 2*nM:5*nM//2], 0] * avec[0,0] + disp_coords[np.r_[nM//2:nM, 2*nM:5*nM//2], 1] * avec[1,0]
yi1 = disp_coords[np.r_[nM//2:nM, 2*nM:5*nM//2], 0] * avec[0,1] + disp_coords[np.r_[nM//2:nM, 2*nM:5*nM//2], 1] * avec[1,1]
zi1 = disp_coords[np.r_[nM//2:nM, 2*nM:5*nM//2], 2] * avec[2,2]



x2c = x2*avec[0,0] + y2*avec[1,0]
y2c = x2*avec[0,1] + y2*avec[1,1]

x2cbig, y2cbig, z2cbig = [], [], []
for i in range(-5,6):
    for j in range(-5,6):
        shift = i*avec[0] + j*avec[1]
        x2cbig.append(x2c+shift[0])
        y2cbig.append(y2c+shift[1])
        z2cbig.append(zi2)

x2a = np.concatenate(x2cbig)
y2a = np.concatenate(y2cbig)
z2a = np.concatenate(z2cbig)    



x1c = x1*avec[0,0] + y1*avec[1,0]
y1c = x1*avec[0,1] + y1*avec[1,1]

x1cbig, y1cbig, z1cbig = [], [], []
for i in range(-5,6):
    for j in range(-5,6):
        shift = i*avec[0] + j*avec[1]
        x1cbig.append(x1c+shift[0])
        y1cbig.append(y1c+shift[1])
        z1cbig.append(zi1)

x1a = np.concatenate(x1cbig)
y1a = np.concatenate(y1cbig)
z1a = np.concatenate(z1cbig)


xmin = np.min([x2a,x1a])
xmax = np.max([x2a,x1a])
ymin = np.min([y2a,y1a])
ymax = np.max([y2a,y1a])

grid_x = np.linspace(xmin, xmax, nr, endpoint=True)
grid_y = np.linspace(ymin, ymax, nr, endpoint=True)
Xa, Ya = np.meshgrid(grid_x, grid_y)

Zi2 = griddata((x2a, y2a), z2a, (Xa, Ya), method='nearest')

Zi1 = griddata((x1a, y1a), z1a, (Xa, Ya), method='nearest')

Za = Zi2 - Zi1 + (0.6166108331837824-0.3833891668163061)*30.6301151979793111


fig, ax = plt.subplots(figsize=(6,6))

levels = np.linspace(Za.min(), Za.max(), 101)
cf = ax.contourf(Xa, Ya, Za, levels=levels, cmap='viridis')
cbar = plt.colorbar(cf)
cbar.formatter = mtick.FormatStrFormatter('%.2f')
cbar.update_ticks()
cbar.ax.tick_params(labelsize=fs)

ax.quiver(0, 0, avec[0,0], avec[0,1], color='black', scale=1, scale_units='xy', width=0.005)
ax.quiver(0, 0, avec[1,0], avec[1,1], color='black', scale=1, scale_units='xy', width=0.005)
ax.text(avec[0,0] + 0.5, avec[0,1], r'$L_1$', fontsize=fs, color='black')
ax.text(avec[1,0] + 0.5, avec[1,1], r'$L_2$', fontsize=fs, color='black')


margin = 70
ax.set_xlim(-margin, margin)
ax.set_ylim(-margin, margin)
ax.set_aspect('equal')
ax.set_title(r'$h^{-}$(Å)',fontsize=fs)
ax.set_xlabel('x(Å)',fontsize=fs)
ax.set_ylabel('y(Å)',fontsize=fs)
ax.set_xticks([-50,0,50])
ax.set_xticklabels(['-50','0','50'],fontsize=fs)
ax.set_yticks([-50,0,50])
ax.set_yticklabels(['-50','0','50'],fontsize=fs)
plt.savefig('h_minus.png', dpi=600, bbox_inches='tight')
plt.savefig('h_minus.pdf', format='pdf', dpi=600, bbox_inches='tight')
plt.show()
