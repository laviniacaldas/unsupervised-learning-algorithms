import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import numpy as np


class agrupamento:
  # Construtor da classe kmeans:
    def __init__(self, dados):
       self.dados = dados
       self.K = int
       self.categoria = np.array([])

    def kmeans(self, K):
        self.K = K
        '''
        1) Sortear K pontos dentro dos dados para inicializacão
        '''
        id = np.random.permutation(np.size(self.dados,0))
        id = id[0:self.K].T
        centro = self.dados[id,:]

        '''
        2) Calcular distância de cada amostra para todos os centros e identificar
        aquele que está mais próximo
        '''
        distancia = cdist(centro,self.dados)

        self.categoria = distancia.argmin(0)
        dist_min = np.amin(distancia, axis = 0)


        '''
        3) Atualizar centro e repetir o processo
        '''
        condicao = 1
        epsilon = 1e-5
        newCentro = np.zeros((self.K,np.size(centro,1)))

        while (condicao):
            for id in range(self.K):
                newCentro[id, :] = np.mean(self.dados[self.categoria == id,:], axis=0)
            
            distanciaCentros = np.diag(cdist(newCentro,centro))
            if np.amin(distanciaCentros) < epsilon:
                condicao = 0
            
            centro = newCentro
            distancia = cdist(centro,self.dados)
            self.categoria = distancia.argmin(0)
            dist_min = np.amin(distancia, axis = 0)
        
        return centro, self.categoria
    
    def plot(self, centro, categoria):
        plt.figure(1)
        plt.plot(self.dados[:,0], self.dados[:,1], 'b.')
        plt.plot(centro[:,0], centro[:,1], 'xr')

        plt.figure(2)
        colormap = np.array(['r', 'g', 'b', 'c', 'y'])
        plt.scatter(self.dados[:,0], self.dados[:,1], s=3,c=colormap[categoria])
        plt.plot(centro[:,0], centro[:,1], 'xk')
        plt.show()


