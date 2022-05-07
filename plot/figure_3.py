
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from openTSNE import TSNE



atom_0_0 = pd.read_csv('E:/0-Ivos_09151445-sencond pro/6-Cal_res/0-from_right/0903/mid_out/set_atom_60_0.csv', header=None, sep=",")
atom_0_1 = pd.read_csv('E:/0-Ivos_09151445-sencond pro/6-Cal_res/0-from_right/0903/mid_out/set_bond_60_0.csv', header=None, sep=",")
atom_0_2 = pd.read_csv('E:/0-Ivos_09151445-sencond pro/6-Cal_res/0-from_right/0903/mid_out/lstm_60_0.csv', header=None, sep=",")

atom_0_3 = pd.read_csv('E:/0-Ivos_09151445-sencond pro/6-Cal_res/0-from_right/1110/1109/mid_out/lstm_10_0.csv', header=None, sep=",")

mat_com = pd.read_csv('E:/0-Ivos_09151445-sencond pro/2-CODE/0-Models/3-plot_model/datasets/comp.csv', header=None, sep=",")
bandgap = pd.read_csv('./final_targets_1.csv', header=None, sep=",")


fe = pd.read_csv('./real_deta_0-0.csv', header=None, sep=",")
fe = np.array(fe).reshape(-1)

F_index = pd.read_csv('./F_index.csv', header=None, sep=",")
Si_index = pd.read_csv('./Si_index.csv', header=None, sep=",")
Br_index = pd.read_csv('./Br_index.csv', header=None, sep=",")

F_index = np.array(F_index,dtype = 'int').reshape(-1)
Si_index = np.array(Si_index,dtype = 'int').reshape(-1)
Br_index = np.array(Br_index,dtype = 'int').reshape(-1)

cap_loss = pd.read_csv('./cap_loss.csv', header=None, sep=",")
cap_loss = np.array(cap_loss).reshape(-1) + 0.1

sym_id = pd.read_csv('./sym_id_1.csv', header=None, sep=",")
sym_id = np.array(sym_id).reshape(-1)

sg_id = pd.read_csv('./all_spacegroup_id_1.csv', header=None, sep=",")
sg_id = np.array(sg_id).reshape(-1)

mat_cs_is = pd.read_csv('./all_cs_id.csv', header=None, sep=",")
mat_cs_is = np.array(mat_cs_is)

cs = mat_cs_is[:, 1].reshape(-1)

atom_0_0 = np.array(atom_0_0)
atom_0_1 = np.array(atom_0_1)
atom_0_2 = np.array(atom_0_2)
atom_0_3 = np.array(atom_0_3)

mat_com = np.array(mat_com).reshape(-1)
bandgap = np.array(bandgap).reshape(-1) + 0.5

cmap1 = sns.diverging_palette(200,20,sep=20, s=75, l=70,as_cmap=True)


color1 = ['tomato', 'tomato', 'tomato', 'tomato', 'tomato', 'tomato', 'tomato', 'tomato', 'tomato', 'tomato', 'tomato',
          'darkorange', 'darkorange', 'darkorange', 'darkorange', 'darkorange', 'darkorange', 'darkorange', 'darkorange', 'darkorange', 'darkorange', 'darkorange', 'darkorange', 'darkorange',
          'seagreen', 'seagreen','seagreen','seagreen','seagreen','seagreen','seagreen','seagreen','seagreen',
          'darkturquoise', 'darkturquoise', 'darkturquoise', 'darkturquoise', 'darkturquoise', 'darkturquoise', 'darkturquoise', 'darkturquoise', 'darkturquoise',
          'teal', 'teal', 'teal', 'teal', 'teal', 'teal', 'teal',
          'blueviolet','blueviolet','blueviolet','blueviolet']

color_1 = ['tomato']
color_2 = ['darkorange']
color_3 = ['seagreen']
color_4 = ['darkturquoise']
color_5 = ['teal']


Al_O_index = pd.read_csv('./Al_O_index.csv', header=None, sep=",")
Ba_O_index = pd.read_csv('./Ba_O_index.csv', header=None, sep=",")
Ca_O_index = pd.read_csv('./Ca_O_index_1.csv', header=None, sep=",")
K_O_index = pd.read_csv('./K_O_index.csv', header=None, sep=",")
Sr_O_index = pd.read_csv('./Sr_O_index.csv', header=None, sep=",")


Al_O_index = np.array(Al_O_index,dtype = 'int').reshape(-1)
Ba_O_index = np.array(Ba_O_index,dtype = 'int').reshape(-1)
Ca_O_index = np.array(Ca_O_index,dtype = 'int').reshape(-1)
K_O_index = np.array(K_O_index,dtype = 'int').reshape(-1)
Sr_O_index = np.array(Sr_O_index,dtype = 'int').reshape(-1)

mat_index = np.hstack((Al_O_index, Ba_O_index, Ca_O_index, K_O_index, Sr_O_index))
color_map = np.hstack((color_1 * len(Al_O_index), color_2 * len(Ba_O_index), color_3 * len(Ca_O_index),
                       color_4 * len(K_O_index), color_5 * len(Sr_O_index)))

tsne = TSNE(n_components=2, perplexity=8, n_iter=30,
            metric="euclidean", n_jobs=1, random_state=42)


#mat_atom_embedding = tsne.fit(atom_0_0)
#mat_bond_embedding = tsne.fit(atom_0_1)
mat_che_embedding = tsne.fit(atom_0_3)

#mat_atom = np.array(mat_atom_embedding)
#mat_bond = np.array(mat_bond_embedding)
mat_che = np.array(mat_che_embedding)


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(24, 20))

plt.scatter(x = mat_che[:, 0], y = mat_che[:, 1],
            alpha=0.4, s = 500 * fe, c = 150 * fe)

ax.set(ylabel = 'Chemical Environment T-SNE-2', xlabel = 'Chemical Environment T-SNE-1')
plt.yticks(fontproperties = 'Helvetica',size = 28)
plt.xticks(fontproperties = 'Helvetica', size = 28)

ax = plt.gca()
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
#plt.xticks(fontproperties='Calibri', size=32)
#plt.yticks((-9, 0, 9), fontproperties='Calibri', size=28,rotation = 0)
#plt.xticks((-10, 0, 10), fontproperties='Calibri', size=28,rotation = 0)
#plt.ylim((-9, 9))
#plt.xlim((-10, 10))
ax.legend(loc = 'lower right', frameon = False,prop={'family' : 'Helvetica', 'size' : 32})
ax1 = plt.gca()
ax1.spines['bottom'].set_linewidth(2)
ax1.spines['left'].set_linewidth(2)
ax1.spines['right'].set_linewidth(2)
ax1.spines['top'].set_linewidth(2)


atom_embedding = tsne.fit(atom_0_0[:5600])
bond_embedding = tsne.fit(atom_0_1[:5600])
che_embedding = tsne.fit(atom_0_2[:5600])

total_n = 5600
index = np.arange(total_n)
mol_index = np.random.permutation(index)
index_part = index[:500]
#embedding.optimize(n_iter=500, exaggeration=12, momentum=0.5, inplace=True)
#embedding.optimize(n_iter=750, momentum=0.8, inplace=True)
#utils.plot(embedding, sg_id[:5600], s = 70)
#plt.show()

#embedding_all = np.array(embedding)

t_sne_x = np.array(che_embedding[Ca_O_index])[:, 0]
t_sne_y = np.array(che_embedding[Ca_O_index])[:, 1]

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 16))
plt.scatter(x = t_sne_x, y = t_sne_y, alpha=0.4, s = 100 * bandgap[Ca_O_index], cmap='greys', label = 'atom-bond')
#plt.scatter(x = t_sne_x, y = t_sne_y, alpha=0.5, s = 100 * bandgap[:5600], c = 20*sg_id[:5600], label = 'atom-bond')

for i in range(len(Ca_O_index)):
    plt.annotate(mat_com[Ca_O_index[i]], xy = (t_sne_x[i], t_sne_y[i]),
                 xytext = (t_sne_x[i]-0.15, t_sne_y[i]-0.055),fontsize=12)

ax.set(ylabel = 'Bond-T-SNE', xlabel = 'Che-T-SNE')
plt.yticks(fontproperties = 'Helvetica',size = 24)
plt.xticks(fontproperties = 'Helvetica', size = 24)

plt.colorbar()
plt.show()


print("SUCCESSFUL")