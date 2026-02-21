import pandas as pd
import numpy as np
import agrupamento as gp
import GMM as GMM

data = pd.read_table("brincos.txt", sep=r"\s+", header=None)

idx_a = (data[2]==1)
idx_b = (data[2]==-1)

dados = np.c_[data[0][idx_a], data[1][idx_a]]

numGrupos = 5
G = gp.agrupamento(dados)
[centro, categoria] = G.kmeans(numGrupos)
#G.plot(centro, categoria)

modelo = GMM.GMM(dados, categoria, numGrupos)
w, media, matrizCovariancia, logVerossimilhanca = modelo.fit(centro, 15)

modelo.plot(media)
modelo.plotAprendizado(logVerossimilhanca)
