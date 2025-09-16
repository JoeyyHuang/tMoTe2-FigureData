import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

fs = 12

x = [2.13, 2.28, 2.45, 2.65, 2.88, 3.15, 3.48, 3.89, 4.41, 5.09, 6.01, 7.34, 9.43]
y = [7.080156870564469, 7.085230421336841, 7.091111885261086,7.098018343016108, 7.106248532701526, 7.116211895606476, 7.128449965648899, 7.143606507500495, 7.162237728353549, 7.184324643898291, 7.208690520827003, 7.233127103081719, 7.256096202099154]

fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(x, y, color='red', marker='o', s=50)
line, = ax.plot(x, y, color='blue', linestyle='--', linewidth=1)

#ax.set_title('interlayer distance')
ax.set_xlabel(r'$\theta$(Deg)',fontsize=fs)
ax.set_ylabel(r'$d_0$(Å)',fontsize=fs)

ax.tick_params(axis='x', labelsize=fs)
ax.tick_params(axis='y', labelsize=fs)
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))

plt.savefig('d0_angle.png', dpi=600, bbox_inches='tight')
plt.savefig('d0_angle.pdf', format='pdf', dpi=600, bbox_inches='tight')
plt.show()
