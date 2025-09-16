import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

alat = 3.5210297778562545
ms4=8

num_kpt_band = 15
nbnd_fit = 2
Nr = 21
num_k1C = 12
nbnd0 = 242

# DFT relaxed band
bd=np.load(f'DFT_relaxed_band_{ms4}.npy')
bandtar_ms4=np.zeros((nbnd_fit,num_kpt_band))
for i in range(nbnd_fit):
    bandtar_ms4[i,:]=bd[:,i]

# DFT unrelax band
bd=np.load(f'DFT_unrelax_band_{ms4}.npy')
bandtar_ms4_us=np.zeros((nbnd_fit,num_kpt_band))
for i in range(nbnd_fit):
    bandtar_ms4_us[i,:]=bd[:,i]

# DFT relaxed charge line
ldos_ref_ms4=np.load(f'DFT_relaxed_charge_line_{ms4}.npy')

# DFT unrelax charge line
ldos_ref_ms4_us=np.load(f'DFT_unrelax_charge_line_{ms4}.npy')


# Model relaxed band
ek_ms4=np.load(f'Model_relaxed_band_{ms4}.npy')

# Model unrelax band
ek_ms4_un=np.load(f'Model_unrelax_band_{ms4}.npy')

# Model relaxed charge
ldos_ms4=np.load(f'Model_relaxed_charge_{ms4}.npy')

# Model unrelax charge
ldos_ms4_un=np.load(f'Model_unrelax_charge_{ms4}.npy')

# Model relaxed charge line
ldos_line_ms4=np.load(f'Model_relaxed_charge_line_{ms4}.npy')

# Model unrelax charge line
ldos_line_ms4_un=np.load(f'Model_unrelax_charge_line_{ms4}.npy')

def la_ve(Ls):
    avec = np.array([[np.sqrt(3)/2,-1/2],[np.sqrt(3)/2,1/2]])*Ls
    bvec = np.array([[1/np.sqrt(3),-1],[1/np.sqrt(3),1]])*2*np.pi/Ls

    G = np.array([0,0])
    M = np.array([0.5,0.5])
    K = np.array([2/3,1/3])

    kvecb = np.zeros((num_kpt_band,2))

    delta = M - G
    for i in range(5):
        kvecb[i] = delta*i/5 + G

    delta = K - M
    for i in range(3):
        kvecb[i+5] = delta*i/3 + M

    delta = G - K
    for i in range(6):
        kvecb[i+8] = delta*i/6 + K

    kvecb[num_kpt_band-1] = kvecb[0]

    klen = np.zeros(num_kpt_band)

    for ik in range(num_kpt_band):
        kvecb[ik] = np.dot(kvecb[ik],bvec)
        if ik > 0:
            klen[ik] = klen[ik-1] + np.linalg.norm(kvecb[ik]-k0)
        k0 = kvecb[ik]

    N = 8
    kvecb2 = []

    delta = M - G
    for i in range(5*N):
        k = delta*i/(5*N) + G
        kvecb2.append(k)

    delta = K - M
    for i in range(3*N):
        k = delta*i/(3*N) + M
        kvecb2.append(k)

    delta = G - K
    for i in range(6*N):
        k = delta*i/(6*N) + K
        kvecb2.append(k)

    kvecb2.append([0,0])

    kvecb2 = np.array(kvecb2)

    klen2 = np.zeros(14*N+1)

    for ik in range(14*N+1):
        kvecb2[ik] = np.dot(kvecb2[ik],bvec)
        if ik > 0:
            klen2[ik] = klen2[ik-1] + np.linalg.norm(kvecb2[ik]-k0)
        k0 = kvecb2[ik]

    rc = np.zeros((Nr*Nr,2))
    for r1 in range(0,Nr):
        for r2 in range(0,Nr):
            rc[r1*Nr+r2,0:2]= (r1-(Nr-1)//2)*avec[0]/Nr+(r2-(Nr-1)//2)*avec[1]/Nr
    return klen, klen2, avec, rc

Ls_ms4 = np.sqrt(3 * ms4**2 + 3 * ms4 + 1) * alat
kline_ms4,klen_ms4,avec_ms4,rc_ms4=la_ve(Ls_ms4)

def plot_band(ms, kline_fine, ek_fine, kline_in,  bandtar_in, relax_indx):
    fs = 18
    plt.ylabel('E (eV)',fontsize=fs)
    plt.xticks([kline_in[0],kline_in[5],kline_in[8],kline_in[14]],[chr(915), 'M', 'K', chr(915)],fontsize=fs)
    plt.xlim(kline_in[0],kline_in[14])
    plt.ylim(-0.1,0.01)
    plt.yticks(fontsize=fs)

    for j in range(5):
        if j == 0:
            plt.plot(kline_fine, ek_fine[:, j], 'b', label='Model')
        else:
            plt.plot(kline_fine, ek_fine[:, j], 'b')

    plt.scatter(kline_in, bandtar_in[0], color='red', label='DFT')
    for j in range(1, nbnd_fit):
        plt.scatter(kline_in, bandtar_in[j], color='red')

    plt.legend(loc='lower right',frameon=False,fontsize=fs)

    cos_theta = (3 * ms**2 + 3 * ms + 0.5) / (3 * ms**2 + 3 * ms + 1)
    theta = np.degrees(np.arccos(cos_theta))

    if relax_indx==0:
        plt.title(rf'Band structure, Non-relaxed, ${theta:.2f}^\circ$',fontsize=fs)
        plt.savefig(f'Band_Non-relaxed_{theta:.2f}°.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'Band_Non-relaxed_{theta:.2f}°.pdf', format='pdf', dpi=300, bbox_inches='tight')
    elif relax_indx==1:
        plt.title(rf'Band structure, Relaxed, ${theta:.2f}^\circ$',fontsize=fs)
        plt.savefig(f'Band_Relaxed_{theta:.2f}°.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'Band_Relaxed_{theta:.2f}°.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.close()

    E_error = 0
    for_check = np.zeros(len(kline_in)-1,dtype=int)
    for ik in range(len(kline_in)-1):
        for jk in range(len(kline_fine)-1):
            if abs(kline_in[ik]-kline_fine[jk]) < 1e-5:
                for_check[ik] = 100
                E_error += abs(bandtar_in[0,ik]-ek_fine[jk,4])
                E_error += abs(bandtar_in[1,ik]-ek_fine[jk,3])
    if np.any(for_check == 0):
        print('error: some k point not match')
    E_error = E_error/(len(kline_in)-1)/2
    band_width = abs(bandtar_in[0].max()-bandtar_in[1].min())
    percent = E_error/band_width
    if relax_indx==0:
        print(f'energy error on average, Non-relaxed, {theta:.2f}° : {E_error:.12f} eV')
        print(f'energy error percentage, Non-relaxed, {theta:.2f}° : {percent*100:.12f}%')
    elif relax_indx==1:
        print(f'energy error on average, Relaxed, {theta:.2f}° : {E_error:.12f} eV')
        print(f'energy error percentage, Relaxed, {theta:.2f}° : {percent*100:.12f}%')
    print()

plot_band(ms4, klen_ms4, ek_ms4_un, kline_ms4, bandtar_ms4_us, 0)

plot_band(ms4, klen_ms4, ek_ms4, kline_ms4, bandtar_ms4, 1)


def plot_chgden(ms,ldos,avec_ms,rc_ms,relax_indx):

    fs = 18
    na1 = 6
    na2 = 6
    X, Y, Z = [], [], []
    for i in range(-na1, na1+1):
        for j in range(-na2, na2+1):
            shift = i*avec_ms[0] + j*avec_ms[1]
            xyt = rc_ms + shift
            X.append(xyt[:, 0])
            Y.append(xyt[:, 1])
            Z.append(ldos)

    X = np.concatenate(X)
    Y = np.concatenate(Y)
    Z = np.concatenate(Z)

    plt.figure()
    plt.gca().set_aspect('equal')
    cnt = plt.tricontourf(X, Y, Z, levels=50, cmap='viridis')
    cbar = plt.colorbar(cnt, format='%.4f', ticks=np.linspace(0, Z.max(), 5))
    cbar.ax.tick_params(labelsize=fs)

    plt.xlim(-40,40)
    plt.ylim(-40,40)
    plt.xticks([-40,-20,0,20,40],['-40','-20','0','20','40'],fontsize=fs)
    plt.yticks([-40,-20,0,20,40],['-40','-20','0','20','40'],fontsize=fs)

    Ls = np.linalg.norm(avec_ms[0])
    R = Ls / np.sqrt(3)
    theta = np.linspace(0, 2*np.pi, 7)[:-1]
    vx = R * np.cos(theta)
    vy = R * np.sin(theta)

    ws_vertices = np.c_[vx, vy]
    hex_patch = Polygon(ws_vertices,closed=True,edgecolor='black',facecolor='none',linewidth=1.5)
    plt.gca().add_patch(hex_patch)

    plt.text(vx[0], vy[0], 'XM', ha='center', va='center', color='white', fontsize=fs)
    plt.text(vx[3], vy[3], 'MX', ha='center', va='center', color='white', fontsize=fs)
    plt.text(0, 0, 'MM', ha='center', va='center', color='white', fontsize=fs)

    plt.xlabel('x(Å)',fontsize=fs)
    plt.ylabel('y(Å)',fontsize=fs)

    #ldosp=np.zeros((3,Nr,Nr))
    #for n0 in range(Nr):
    #    for n1 in range(Nr):
    #        ldosp[0:2,n0,n1]=rc_ms[n0*Nr+n1,0:2]
    #        ldosp[2,n0,n1]=ldos[n0*Nr+n1]
    #plt.axis('equal')
    #plt.contourf(ldosp[0],ldosp[1],ldosp[2],levels=50)
    #plt.colorbar()

    cos_theta = (3 * ms**2 + 3 * ms + 0.5) / (3 * ms**2 + 3 * ms + 1)
    theta = np.degrees(np.arccos(cos_theta))

    if relax_indx==0:
        plt.title(rf'Charge density, Non-relaxed, ${theta:.2f}^\circ$',pad=20,fontsize=fs)
        plt.savefig(f'Charge_Non-relaxed_{theta:.2f}°.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'Charge_Non-relaxed_{theta:.2f}°.pdf', format='pdf', dpi=300, bbox_inches='tight')
    elif relax_indx==1:
        plt.title(rf'Charge density, Relaxed, ${theta:.2f}^\circ$',pad=20,fontsize=fs)
        plt.savefig(f'Charge_Relaxed_{theta:.2f}°.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'Charge_Relaxed_{theta:.2f}°.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.close()

plot_chgden(ms4, ldos_ms4_un, avec_ms4, rc_ms4, 0)

plot_chgden(ms4, ldos_ms4, avec_ms4, rc_ms4, 1)


def plot_chgden_line(ms,ldos,ldos_ref,relax_indx):
    fs = 18
    numr = 21
    Nline = 63
    ldosp=np.zeros((2,Nline))
    for n0 in range(Nline):
        for n1 in range(1):
            ldosp[0,n0]=n0
            ldosp[1,n0]=ldos[n0]
    plt.xlim(0,Nline)
    plt.locator_params(axis='y', nbins=3, steps=None)
    plt.plot(ldosp[0],ldosp[1],color='blue',label='Model')

    x_scat = np.zeros(numr, dtype=int)
    for i in range(numr):
        for j in range(Nline):
            if (i-(numr-1)//2)/numr == (j-(Nline-1)//2)/Nline:
                x_scat[i] = j
    plt.scatter(x_scat,ldos_ref.real/9,color='red',label='DFT')
    plt.ylabel('charge density',fontsize=fs)
    plt.xticks([ldosp[0,10],ldosp[0,31],ldosp[0,52]],['MX','MM','XM'],fontsize=fs)
    plt.yticks(fontsize=fs)
    if relax_indx == 0:
        plt.legend(loc='upper right', frameon=False, fontsize=fs)
    if relax_indx == 1:
        plt.legend(loc='upper center', frameon=False, fontsize=fs)

    cos_theta = (3 * ms**2 + 3 * ms + 0.5) / (3 * ms**2 + 3 * ms + 1)
    theta = np.degrees(np.arccos(cos_theta))

    if relax_indx==0:
        plt.savefig(f'Line_Non-relaxed_{theta:.2f}°.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'Line_Non-relaxed_{theta:.2f}°.pdf', format='pdf', dpi=300, bbox_inches='tight')
    elif relax_indx==1:
        plt.savefig(f'Line_Relaxed_{theta:.2f}°.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'Line_Relaxed_{theta:.2f}°.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.close()

    D_error = 0
    for i in range(numr):
        D_error += abs(ldos_ref[i].real/9-ldosp[1,x_scat[i]])
    D_error = D_error/numr
    band_width = abs(ldos_ref.real.max()/9-ldos_ref.real.min()/9)
    percent = D_error/band_width
    if relax_indx==0:
        print(f'density error on average, Non-relaxed, {theta:.2f}° : {D_error:.12f}')
        print(f'density error percentage, Non-relaxed, {theta:.2f}° : {percent*100:.12f}%')
    elif relax_indx==1:
        print(f'density error on average, Relaxed, {theta:.2f}° : {D_error:.12f}')
        print(f'density error percentage, Relaxed, {theta:.2f}° : {percent*100:.12f}%')
    print()


plot_chgden_line(ms4, ldos_line_ms4_un, ldos_ref_ms4_us, 0)

plot_chgden_line(ms4, ldos_line_ms4, ldos_ref_ms4, 1)
