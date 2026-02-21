import numpy as np
import matplotlib.pyplot as plt

class GMM():
  def __init__(self, dados, categoria, K):
    self.dados = dados
    self.categoria = categoria
    self.K = K

  def fit(self,centro, ite_MAX):
    D = self.dados
    '''
    1) Inicializacao dos paramentros da guassiana
    '''
    u = centro
    numDim = np.size(self.dados,1)
    E = np.zeros((numDim,numDim,self.K))

    for id in range(self.K):
      g = self.dados[self.categoria == id,:]
      E[:,:,id ]= (1/(np.size(g,0)))*(g - u[id,:]).T@(g - u[id,:])

    w = abs(np.random.randn(self.K))
    w = w/np.sum(w)
    u = u.T
    
    ite_max = ite_MAX
    gamma = np.zeros((self.K,np.size(D,0)))
    LV = np.zeros((ite_max,))
    
    for ite in range(ite_max):

      #for id in range(self.K):
        #A,V = np.linalg.eig(E[:,:,id])
        #E[:,:,id] = V@np.abs(A)@V.T
        
      for i in range(self.K):
        for j in range(np.size(D,0)):
          gamma[i,j] = w[i]*(1/(2*np.pi*np.sqrt(np.linalg.det(E[:,:,i])))) * np.exp(-0.5*( ((D[j,:].T-u[:,i]).T)@np.linalg.inv(E[:,:,i]+0.1)@(D[j,:].T-u[:,i])))
        

      LV[ite] = np.sum(np.log(np.sum(gamma,axis = 0)))
      gammau = gamma/np.sum(gamma,axis = 0)
      soma = np.sum(gammau,axis = 1)

      
      w = soma/sum(soma)

      for i in range(self.K):
        E[:,:,i] = (D - u[:,i].T).T@np.diag(gammau[i,:])@(D - u[:,i].T)/soma[i]


      u = ((gammau@D)/np.expand_dims(soma,axis=1)).T
    
    return w, u, E, LV

  def plot(self, media):
    plt.figure(2)
    plt.plot(media[0,:], media[1,:], 'xr')
    plt.plot(self.dados[:,0], self.dados[:,1], '.b')
    plt.show()
    

  def plotAprendizado(self, logVerossimilhanca):
    plt.figure(1)
    plt.plot(logVerossimilhanca)
    plt.show()



