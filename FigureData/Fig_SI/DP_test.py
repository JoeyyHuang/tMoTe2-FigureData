import numpy as np
import matplotlib.pyplot as plt

fs = 10

fig = plt.figure(figsize=(6,6),dpi=600)

data1 = np.loadtxt('dp_dft.e_peratom.out', comments='#', usecols=(0, 1))
x1 = data1[:, 0]
y1 = data1[:, 1]
ax1 = fig.add_subplot(221)
ax1.set_xlabel(r'E$_{\mathrm{DFT}}$ (eV)',fontsize=fs)
ax1.set_ylabel(r'E$_{\mathrm{DP}}$ (eV)',fontsize=fs)
ax1.set_xlim(-6.38,-6.18)
ax1.set_ylim(-6.38,-6.18)
ax1.set_aspect('equal')
ax1.set_yticks([-6.35,-6.30,-6.25,-6.20])
ax1.set_yticklabels(['-6.35','-6.30','-6.25','-6.20'],fontsize=fs)
ax1.tick_params(axis='both', labelsize=fs)
kw = dict(transform=ax1.transAxes, fontsize=fs, va='top', ha='left')
ax1.text(-0.2, 1.05, '(a)', **kw)
ax1.scatter(x1, y1, color='blue', marker='o', s=1, label='energy per atom')
ax1.plot([-6.38, -6.18], [-6.38, -6.18], linestyle='--', color='black', linewidth=0.8)
ax1.legend(loc='upper left', fontsize=fs, frameon=False)
print('E maximum error:',max(abs(x1-y1)),' eV')

data2 = np.loadtxt('dp_dft.f.out', comments='#', usecols=(0, 3))
x2 = data2[:, 0]
y2 = data2[:, 1]
ax2 = fig.add_subplot(222)
ax2.set_xlabel(r'F$_{\mathrm{DFT}}$ (eV/Å)',fontsize=fs)
ax2.set_ylabel(r'F$_{\mathrm{DP}}$ (eV/Å)',fontsize=fs)
ax2.set_xlim(-1.75,1.75)
ax2.set_ylim(-1.75,1.75)
ax2.set_aspect('equal')
ax2.tick_params(axis='both', labelsize=fs)
kw.update(transform=ax2.transAxes)
ax2.text(-0.2, 1.05, '(b)', **kw)
ax2.scatter(x2, y2, color='blue', marker='o', s=1, label='atomic force Fx')
ax2.plot([-1.75, 1.75], [-1.75, 1.75], linestyle='--', color='black', linewidth=0.8)
ax2.legend(loc='upper left', fontsize=fs, frameon=False)
print('Fx maximum error:',max(abs(x2-y2)),' eV/Å')

data3 = np.loadtxt('dp_dft.f.out', comments='#', usecols=(1, 4))
x3 = data3[:, 0]
y3 = data3[:, 1]
ax3 = fig.add_subplot(223)
ax3.set_xlabel(r'F$_{\mathrm{DFT}}$ (eV/Å)',fontsize=fs)
ax3.set_ylabel(r'F$_{\mathrm{DP}}$ (eV/Å)',fontsize=fs)
ax3.set_xlim(-1.75,1.75)
ax3.set_ylim(-1.75,1.75)
ax3.set_aspect('equal')
ax3.tick_params(axis='both', labelsize=fs)
kw.update(transform=ax3.transAxes)
ax3.text(-0.2, 1.05, '(c)', **kw)
ax3.scatter(x3, y3, color='blue', marker='o', s=1, label='atomic force Fy')
ax3.plot([-1.75, 1.75], [-1.75, 1.75], linestyle='--', color='black', linewidth=0.8)
ax3.legend(loc='upper left', fontsize=fs, frameon=False)
print('Fy maximum error:',max(abs(x3-y3)),' eV/Å')

data4 = np.loadtxt('dp_dft.f.out', comments='#', usecols=(2, 5))
x4 = data4[:, 0]
y4 = data4[:, 1]
ax4 = fig.add_subplot(224)
ax4.set_xlabel(r'F$_{\mathrm{DFT}}$ (eV/Å)',fontsize=fs)
ax4.set_ylabel(r'F$_{\mathrm{DP}}$ (eV/Å)',fontsize=fs)
ax4.set_xlim(-3,3)
ax4.set_ylim(-3,3)
ax4.set_aspect('equal')
ax4.tick_params(axis='both', labelsize=fs)
kw.update(transform=ax4.transAxes)
ax4.text(-0.2, 1.05, '(d)', **kw)
ax4.scatter(x4, y4, color='blue', marker='o', s=1, label='atomic force Fz')
ax4.plot([-3, 3], [-3, 3], linestyle='--', color='black', linewidth=0.8)
ax4.legend(loc='upper left', fontsize=fs, frameon=False)
print('Fz maximum error:',max(abs(x4-y4)),' eV/Å')

fig.subplots_adjust(hspace=0.4)
fig.subplots_adjust(wspace=0.4)

fig.savefig('DP_test.png',dpi=600,bbox_inches='tight')
plt.savefig('DP_test.pdf', format='pdf', dpi=600, bbox_inches='tight')
plt.show()

