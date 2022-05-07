
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import seaborn as sns


cap_cs_loss = pd.read_csv('./cap_cs_loss.csv', header=None)
com_cs_loss = pd.read_csv('./com_cs_loss.csv', header=None, sep=",")
cs_id = pd.read_csv('./cs_id.csv', header=None, sep=",")


cap_cs_loss = np.array(cap_cs_loss).reshape(-1)
com_cs_loss = np.array(com_cs_loss).reshape(-1)
cs_id = np.array(cs_id).reshape(-1)

b = dict(Counter(cs_id))

#triclinic (1-2)
#monoclinic (3-15)
#orthorhombic (16-74)
#tetragonal (76-141)
#trigonal (143-167)
#hexagonal (173-194)
#cubic (198-229)
cmap1 = sns.diverging_palette(200,50,sep=20, s=75, l=70,as_cmap=True)


cap_triclinic = []
cap_monoclinic = []
cap_orthorhombic = []
cap_tetragonal = []
cap_trigonal = []
cap_hexagonal = []
cap_cubic = []

com_triclinic = []
com_monoclinic = []
com_orthorhombic = []
com_tetragonal = []
com_trigonal = []
com_hexagonal = []
com_cubic = []

# triclinic (1-2)    monoclinic (3-15)   orthorhombic (16-74) tetragonal (76-141)
# trigonal (143-167) hexagonal (173-194) cubic (198-229)

for i in cs_id:
    if i < 3:
        cap_triclinic.append(cap_cs_loss[i])
        com_triclinic.append(com_cs_loss[i])
    elif i< 16:
        cap_monoclinic.append(cap_cs_loss[i])
        com_monoclinic.append(com_cs_loss[i])
    elif i< 75:
        cap_orthorhombic.append(cap_cs_loss[i])
        com_orthorhombic.append(com_cs_loss[i])
    elif i< 142:
        cap_tetragonal.append(cap_cs_loss[i])
        com_tetragonal.append(com_cs_loss[i])
    elif i< 172:
        cap_trigonal.append(cap_cs_loss[i])
        com_trigonal.append(com_cs_loss[i])
    elif i< 197:
        cap_hexagonal.append(cap_cs_loss[i])
        com_hexagonal.append(com_cs_loss[i])
    else:
        cap_cubic.append(cap_cs_loss[i])
        com_cubic.append(com_cs_loss[i])

mae_cap_triclinic = np.mean(np.array(cap_triclinic))
mae_com_triclinic = np.mean(np.array(com_triclinic))

mae_cap_monoclinic = np.mean(np.array(cap_monoclinic))
mae_com_monoclinic = np.mean(np.array(com_monoclinic))

mae_cap_orthorhombic = np.mean(np.array(cap_orthorhombic))
mae_com_orthorhombic = np.mean(np.array(com_orthorhombic))

mae_cap_tetragonal = np.mean(np.array(cap_tetragonal))
mae_com_tetragonal = np.mean(np.array(com_tetragonal))

mae_cap_trigonal = np.mean(np.array(cap_trigonal))
mae_com_trigonal = np.mean(np.array(com_trigonal))

mae_cap_hexagonal = np.mean(np.array(cap_hexagonal))
mae_com_hexagonal = np.mean(np.array(com_hexagonal))

mae_cap_cubic = np.mean(np.array(cap_cubic))
mae_com_cubic = np.mean(np.array(com_cubic))


plt.boxplot(cap_trigonal)
plt.show()


fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(18, 9), sharey=True)
box_1, box_2, box_3, box_4, box_5, box_6 , box_7, box_8, box_9 , box_10, box_11, box_12 , box_13, box_14 = \
    cap_triclinic, com_triclinic, cap_monoclinic, com_monoclinic, cap_orthorhombic, com_orthorhombic,cap_tetragonal, \
    com_tetragonal, cap_trigonal, com_trigonal, cap_hexagonal, com_hexagonal, cap_cubic ,com_cubic
labels = 'O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7'
labels1= 'O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7'


f = plt.boxplot([box_1,  box_3,  box_5, box_7,  box_9, box_11, box_13],
                positions=[1 ,2.5, 4, 5.5, 7, 8.5, 10],
                labels=labels, patch_artist=True, showfliers=False, widths= 0.35,sym = '*')

f1 = plt.boxplot([box_2, box_4, box_6, box_8, box_10, box_12 , box_14],
                 positions=[1.5 ,3, 4.5, 6, 7.5, 9, 10.5],
                 labels=labels1, patch_artist=True, showfliers=False, widths= 0.35, notch = True,sym = '*')

color  = ['red', 'darkorange', 'forestgreen', 'teal', 'royalblue', 'indigo', 'grey']
color1 = ['tomato', 'orange', 'limegreen', 'darkturquoise', 'cornflowerblue', 'blueviolet', 'gray']

for box, c in zip(f['boxes'], color):

    box.set(color=c, linewidth=2)
    box.set(facecolor=c, alpha=0.2)

for box, c in zip(f1['boxes'], color1):

    box.set(color=c, linewidth=2)
    box.set(facecolor=c, alpha=0.2)

for whisker in f['whiskers']:
    whisker.set(color='k', linestyle='--', linewidth=1)
for cap in f['caps']:
    cap.set(color='k', linewidth=1)
for median in f['medians']:
    median.set(color='grey', linewidth=1)

plt.xlabel('Crystal System', fontdict={'family': 'Calibri', 'size': 24})
plt.ylabel('Loss(ev)', fontdict={'family': 'Calibri', 'size': 24})
plt.xticks([1.25 ,2.75, 4.25, 5.75, 7.25, 8.75, 10.25],
           [r'Triclinic', r'Monoclinic', r'Orthorhombic', r'Tetragonal', r'Trigonal', r'Hexagonal', r'Cubic'],
           fontproperties='Calibri', size=20)
plt.yticks([0,0.5,1.0],fontproperties='Calibri',size=18)

ax1 = plt.gca()
ax1.spines['bottom'].set_linewidth(1.5)
ax1.spines['left'].set_linewidth(1.5)
ax1.spines['right'].set_linewidth(1.5)
ax1.spines['top'].set_linewidth(1.5)

plt.rcParams['savefig.dpi'] = 900
plt.rcParams['figure.dpi'] = 900

plt.ylim(-.05, 1.8)

plt.show()


print("SUCCESSFUL")