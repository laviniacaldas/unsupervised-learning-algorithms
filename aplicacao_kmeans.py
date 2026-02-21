import agrupamento as gp
import pandas as pd
import numpy as np

data = pd.read_table("brincos.txt", sep=r"\s+", header=None) #, delim_whitespace=True

idx_a = (data[2]==1)
idx_b = (data[2]==-1)

dados = np.c_[data[0][idx_a], data[1][idx_a]]

numGrupos = 5
G = gp.agrupamento(dados)
[centro, categoria] = G.kmeans(numGrupos)
G.plot(centro, categoria)