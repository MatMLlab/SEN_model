"""
Data Plot
"""

import seaborn as sns
sns.set_theme()

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm
import seaborn as sns
from openTSNE import TSNE


atom_0_0 = pd.read_csv('E:/0-Ivos_09151445-sencond pro/6-Cal_res/0-from_right/0903/mid_out/set_atom_60_0.csv', header=None, sep=",")
atom_0_1 = pd.read_csv('E:/0-Ivos_09151445-sencond pro/6-Cal_res/0-from_right/0903/mid_out/set_bond_60_0.csv', header=None, sep=",")
atom_0_2 = pd.read_csv('E:/0-Ivos_09151445-sencond pro/6-Cal_res/0-from_right/0903/mid_out/lstm_60_0.csv', header=None, sep=",")


mat_com = pd.read_csv('comp.csv', header=None, sep=",")
mat_com = np.array(mat_com).reshape(-1)


atom_inp_1 = pd.read_csv('E:/0-Ivos_09151445-sencond pro/6-Cal_res/0-from_right/0903/mid_out/atom_inp_40_0.csv', header=None, sep=",")
atom_1_0 = pd.read_csv('E:/0-Ivos_09151445-sencond pro/6-Cal_res/0-from_right/0903/mid_out/atom_0_out_60_0.csv', header=None, sep=",")
atom_1_1 = pd.read_csv('E:/0-Ivos_09151445-sencond pro/6-Cal_res/0-from_right/0903/mid_out/atom_1_out_60_0.csv', header=None, sep=",")
atom_1_2 = pd.read_csv('E:/0-Ivos_09151445-sencond pro/6-Cal_res/0-from_right/0903/mid_out/atom_2_out_60_0.csv', header=None, sep=",")
atom_1_3 = pd.read_csv('E:/0-Ivos_09151445-sencond pro/6-Cal_res/0-from_right/0903/mid_out/atom_3_out_60_0.csv', header=None, sep=",")
atom_1_4 = pd.read_csv('E:/0-Ivos_09151445-sencond pro/6-Cal_res/0-from_right/0903/mid_out/atom_4_out_60_0.csv', header=None, sep=",")

atom_len = pd.read_csv('E:/0-Ivos_09151445-sencond pro/2-CODE/0-Models/3-plot_model/datasets/atom_len.csv', header=None, sep=",")
mat_id = pd.read_csv('E:/0-Ivos_09151445-sencond pro/2-CODE/0-Models/3-plot_model/datasets/material_ids.csv', header=None, sep=",")

atom_len = np.array(atom_len).reshape(-1)
mat_id = np.array(mat_id).reshape(-1)

def get_bundary(tar):
    a = 0
    num = 0
    for i in atom_len:
        len = a + i
        if num == tar:
            return a, len
        else:
            a += i
            num += 1

cmap1 = sns.diverging_palette(200,20,sep=20, s=75, l=70,as_cmap=True)

atom_0_0 = np.array(atom_0_0)
atom_0_1 = np.array(atom_0_1)
atom_0_2 = np.array(atom_0_2)

# K-O-?(metal)

atom_seires = np.stack ([atom_0_0[743], atom_0_0[5069], atom_0_0[1493], atom_0_0[4750], atom_0_0[318], atom_0_0[3599],
                         atom_0_0[901], atom_0_0[2836], atom_0_0[3666], atom_0_0[901], atom_0_0[1285], atom_0_0[4235],
                         atom_0_0[2684], atom_0_0[3125], atom_0_0[3844], atom_0_0[670], atom_0_0[1259], atom_0_0[4265],
                         atom_0_0[261], atom_0_0[3189], atom_0_0[896], atom_0_0[4739], atom_0_0[2952], atom_0_0[34],
                         atom_0_0[689], atom_0_0[5453], atom_0_0[238],atom_0_0[3054], atom_0_0[354], atom_0_0[1715],
                         atom_0_0[5375], atom_0_0[1497], atom_0_0[2494], atom_0_0[722], atom_0_0[4618], atom_0_0[1131],
                         atom_0_0[4753], atom_0_0[4545], atom_0_0[2661], atom_0_0[299], atom_0_0[2630], atom_0_0[3978],
                         atom_0_0[5078], atom_0_0[2935], atom_0_0[483], atom_0_0[2938], atom_0_0[4003], atom_0_0[3080],
                         atom_0_0[1961], atom_0_0[3392], atom_0_0[4918], atom_0_0[3345], atom_0_0[628], atom_0_0[1063],
                         atom_0_0[1957], atom_0_0[3955], atom_0_0[5437], atom_0_0[2543], atom_0_0[2523], atom_0_0[260],
                         atom_0_0[5225], atom_0_0[1388], atom_0_0[3743], atom_0_0[5326], atom_0_0[3212]], axis = 0)


txt = [mat_com[743], mat_com[5069], mat_com[1493], mat_com[4750], mat_com[318], mat_com[3599], mat_com[901],
       mat_com[2836], mat_com[3666], mat_com[901], mat_com[1285], mat_com[4235], mat_com[2684], mat_com[3125],
       mat_com[3844], mat_com[670], mat_com[1259], mat_com[4265], mat_com[261], mat_com[3189], mat_com[896],
       mat_com[4739], mat_com[2952], mat_com[34], mat_com[689], mat_com[5453], mat_com[238], mat_com[3054],
       mat_com[354], mat_com[1715], mat_com[5375], mat_com[1497], mat_com[2494], mat_com[722], mat_com[4618],
       mat_com[1131], mat_com[4753], mat_com[4545], mat_com[2661], mat_com[299], mat_com[2630], mat_com[3978],
       mat_com[5078], mat_com[2935], mat_com[483], mat_com[2938], mat_com[4003], mat_com[3080], mat_com[1961],
       mat_com[3392], mat_com[4918], mat_com[3345], atom_0_0[628], mat_com[1063], mat_com[1957], mat_com[3955],
       mat_com[5437], mat_com[2543], mat_com[2523], mat_com[260], mat_com[5225], mat_com[1388], mat_com[3743],
       mat_com[5326], mat_com[3212]]

series = [2005, 5069, 1493, 4750, 318, 3599, 901, 2836, 3666, 901, 1285, 4235, 2684, 3125, 4675, 4734, 1259,
          4265, 261, 3189, 896, 4739, 2952, 34, 689, 5453, 238, 4484, 2703, 1715, 5375, 1497, 2494, 722, 4618,
          1131, 4753, 664, 236, 299, 2630, 3978, 5078, 2935, 483, 2938, 4003, 3080, 1961, 3392, 3299, 3345,
          628, 1063, 1957, 3955, 5437, 2543, 2523, 373, 1400, 1388, 3743, 2113, 3212, 1998]

series1 = [67, 113, 128, 116, 213, 197,
           214, 223, 294, 407, 509, 523,
           603, 751, 833, 857, 904, 915,
           934, 951, 991, 993, 1044, 1023,
           1582, 1245, 1263, 1343, 1348, 1467,
           5375, 1602, 1469, 1744, 1724, 1714,
           1727, 1725, 1806, 2074, 2129, 2280,
           2327, 2355, 2370, 2462, 2452, 2465,
           4244, 4470, 4483, 4593, 628, 4673,
           4715, 4764, 5230, 5302, 4691, 4806,
           4146, 4182, 4167, 4226, 4269]

material = []
for i in series1:
    mat_l, mat_r = get_bundary(i)
    mat = np.array(atom_1_4)[int(mat_l):int(mat_r), :]
    material.append(mat)

Li_l, Li_r = get_bundary(2843)
Li_atom_1 = np.array(atom_1_0)[int(Li_l):int(Li_r), :][0]

H_atom  = np.array(material[0])[4]
Li_atom = np.array(material[1])[0]
Be_atom = np.array(material[2])[0]
B_atom  = np.array(material[3])[1]
C_atom  = np.array(material[4])[20]
N_atom  = np.array(material[5])[-1]
O_atom  = np.array(material[-1])[-1]
F_atom  = np.array(material[6])[-1]
Na_atom = np.array(material[7])[0]
Mg_atom = np.array(material[8])[0]
Al_atom = np.array(material[9])[4] # 10

Si_atom = np.array(material[10])[6]
P_atom  = np.array(material[11])[3]
S_atom  = np.array(material[12])[-1]
Cl_atom = np.array(material[13])[14]
K_atom  = np.array(material[0])[0]
Ca_atom = np.array(material[14])[0]
Sc_atom = np.array(material[15])[0]
Ti_atom = np.array(material[16])[0]
V_atom  = np.array(material[17])[0]
Cr_atom = np.array(material[18])[1]
Mn_atom = np.array(material[19])[0] # 21

Fe_atom = np.array(material[20])[2]
Co_atom = np.array(material[21])[1]
Ni_atom = np.array(material[22])[4]
Cu_atom = np.array(material[23])[2]
Zn_atom = np.array(material[24])[4]
Ga_atom = np.array(material[25])[0]
Ge_atom = np.array(material[26])[4]
As_atom = np.array(material[27])[0]
Se_atom = np.array(material[28])[3]
Br_atom = np.array(material[29])[0] # 31
#Kr_atom = np.array(material[30])[0]

Rb_atom = np.array(material[31])[0]
Sr_atom = np.array(material[32])[3]
Y_atom  = np.array(material[33])[6]
Zr_atom = np.array(material[34])[0]
Nb_atom = np.array(material[35])[0]
Mo_atom = np.array(material[36])[4]
Tc_atom = np.array(material[37])[2]
Ru_atom = np.array(material[38])[4]
Rh_atom = np.array(material[39])[3]
Pd_atom = np.array(material[40])[1]
Ag_atom = np.array(material[41])[1] # 42

Cd_atom = np.array(material[42])[0]
In_atom = np.array(material[43])[4]
Sn_atom = np.array(material[44])[2]
Sb_atom = np.array(material[45])[1]
Te_atom = np.array(material[46])[2]
I_atom  = np.array(material[47])[1]
Xe_atom = np.array(material[48])[0]
Cs_atom = np.array(material[49])[0]
Ba_atom = np.array(material[50])[0]
La_atom = np.array(material[51])[1]

Hf_atom = np.array(material[53])[0] # 53

Ta_atom = np.array(material[54])[0]
W_atom  = np.array(material[55])[2]
Re_atom = np.array(material[56])[0]
Os_atom = np.array(material[57])[1]
Ir_atom = np.array(material[58])[-1]
Pt_atom = np.array(material[59])[4]
Au_atom = np.array(material[60])[2]
Hg_atom = np.array(material[61])[1]
Tl_atom = np.array(material[62])[0]
Pb_atom = np.array(material[63])[4]
Bi_atom = np.array(material[64])[2] # 64

atoms = np.stack([H_atom, Li_atom, Be_atom, B_atom, C_atom, N_atom, O_atom, F_atom, Na_atom, Mg_atom, Al_atom,
                  Si_atom,  P_atom, S_atom, Cl_atom, K_atom, Ca_atom, Sc_atom, Ti_atom, V_atom, Cr_atom, Mn_atom,
                  Fe_atom, Co_atom, Ni_atom, Cu_atom, Zn_atom, Ga_atom, Ge_atom, As_atom, Se_atom, Br_atom, #Kr_atom,
                  Rb_atom, Sr_atom, Y_atom, Zr_atom, Nb_atom, Mo_atom, Tc_atom, Ru_atom, Rh_atom, Pd_atom, Ag_atom,
                  Cd_atom, In_atom,  Sn_atom, Sb_atom, Te_atom, I_atom, Xe_atom, Cs_atom, Ba_atom, La_atom, Hf_atom,
                  Ta_atom, W_atom, Re_atom, Os_atom, Ir_atom, Pt_atom, Au_atom, Hg_atom, Tl_atom, Pb_atom, Bi_atom], axis = 0)

atoms1 = np.stack([H_atom, Li_atom, Na_atom, K_atom, Rb_atom,Cs_atom,
                  Ba_atom, Sr_atom, Ca_atom, Mg_atom, Be_atom,
                  La_atom, Sc_atom, Y_atom, Ti_atom, Zr_atom, Hf_atom,Ta_atom, W_atom, Mo_atom, Re_atom,Tc_atom,Mn_atom,Cr_atom, V_atom,Nb_atom,
                  Fe_atom,Co_atom, Ni_atom,Pd_atom,Pt_atom,Os_atom, Ir_atom, Ru_atom, Rh_atom,
                  Cu_atom, Zn_atom,Ag_atom,Au_atom, Hg_atom,Cd_atom,
                  B_atom, Al_atom, Ga_atom, In_atom, Tl_atom,
                  C_atom, Si_atom, Ge_atom, Sn_atom, Pb_atom,
                  N_atom, P_atom, As_atom, Sb_atom, Bi_atom,
                  O_atom, S_atom, Se_atom, Te_atom,
                  F_atom, Cl_atom, Br_atom,I_atom, Xe_atom,  #Kr_atom,
                  ], axis = 0)


atoms2 = np.stack([Rb_atom,Cs_atom,
                   Ba_atom, Sr_atom, Ca_atom, Mg_atom, Be_atom,
                   La_atom, Sc_atom, Y_atom, Ti_atom,Mn_atom,Cr_atom, V_atom,Nb_atom,
                   Fe_atom,Co_atom, Ni_atom,Pd_atom,Pt_atom,Os_atom, Ir_atom, Ru_atom, Rh_atom,Cu_atom, Zn_atom,Ag_atom,
                    N_atom, P_atom,O_atom, S_atom, Se_atom, Te_atom, F_atom, Cl_atom, Br_atom, I_atom  #Kr_atom,
                  ], axis = 0)

y_label = ['H', 'Li', 'Na', 'K', 'Rb','Cs',
           'Ba', 'Sr', 'Ca', 'Mg', 'Be',
           'La', 'Sc', 'Y', 'Ti', 'Zr', 'Hf','Ta', 'W', 'Mo', 'Re','Tc','Mn','Cr', 'V','Nb',
           'Fe','Co', 'Ni','Pd','Pt','Os', 'Ir', 'Ru', 'Rh',
           'Cu', 'Zn','Ag','Au', 'Hg','Cd',
           'B', 'Al', 'Ga', 'In', 'Tl',
           'C', 'Si', 'Ge', 'Sn', 'Pb',
           'N', 'P', 'As', 'Sb', 'Bi',
           'O', 'S', 'Se', 'Te',
           'F', 'Cl', 'Br','I', 'Xe']

colors = 20 * np.array([ 1,1,1,1,1, 1,
                         2,2,2,2,2,
                         3,3,3,3,3, 3,3,3,3,3, 3,3,3,3,3,
                         4,4,4,4,4, 4,4,4,4,
                         5,5,5,5,5, 5,
                         6,6,6,6,6,
                         7,7,7,7,7,
                         8,8,8,8,8,
                         9,9,9,9,
                         10,10,10,10,10,
                         ])


cmap2 = sns.diverging_palette(330,120,sep=20, s=75, l=70,as_cmap=True)
df_atoms1 = pd.DataFrame(atoms1).T
atoms_corr1 = df_atoms1.corr()

x_label = ['H', 'Li', 'Na', 'K', 'Rb','Cs',
                  'Ba', 'Sr', 'Ca', 'Mg', 'Be',
                  'La', 'Sc', 'Y', 'Ti', 'Zr', 'Hf','Ta', 'W', 'Mo', 'Re','Tc','Mn','Cr', 'V','Nb',
                  'Fe','Co', 'Ni','Pd','Pt','Os', 'Ir', 'Ru', 'Rh',
                  'Cu', 'Zn','Ag','Au', 'Hg','Cd',
                  'B', 'Al', 'Ga', 'In', 'Tl',
                  'C', 'Si', 'Ge', 'Sn', 'Pb',
                  'N', 'P', 'As', 'Sb', 'Bi',
                  'O', 'S', 'Se', 'Te',
                  'F', 'Cl', 'Br','I', 'Xe']

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(36,32))
h = sns.heatmap(atoms_corr1,  xticklabels=x_label, yticklabels=y_label,cbar=False, alpha=1)
cb = h.figure.colorbar(h.collections[0])
cb.ax.tick_params(labelsize=32)
plt.xticks(fontproperties='Calibri', size=32)
plt.yticks(fontproperties='Calibri', size=32,rotation = 0)

#plt.xlabel('Element', fontdict={'family': 'Calibri', 'size': 40})
#plt.ylabel('Element', fontdict={'family': 'Calibri', 'size': 40})

ax1 = plt.gca()
ax1.spines['bottom'].set_linewidth(1.5)
ax1.spines['left'].set_linewidth(1.5)
ax1.spines['right'].set_linewidth(1.5)
ax1.spines['top'].set_linewidth(1.5)

plt.rcParams['savefig.dpi'] = 900
plt.rcParams['figure.dpi'] = 900

plt.show()


print("SUCCESSFUL")