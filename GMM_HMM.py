import numpy as np
import time as time
import GMM as GMM

class GMM_HMM:
    # Construtor da classe HMM:
    def __init__(self, pi, A, w, mi, sigma):
        self.A = A
        self.pi = pi
        self.w = w
        self.mi = mi
        self.sigma = sigma
        self.N = self.A.shape[0]
        
    # Métodos:
    def guassian(self, O, mi, sigma):
        D = np.size(sigma,0)
        c = 1/(((2*np.pi)**(D/2))*((np.linalg.det(sigma))**(1/2)))
        dist = ((O - mi)@np.linalg.inv(sigma))@((O - mi).T)
        return c*np.exp((-1/2)*dist)

    def B(self,O,mi,sigma):
        B = np.zeros((self.N,np.size(O,1)))
        for i in range(self.N):
            for j in range(np.size(O,1)):
                B[i,j] = self.guassian(O[:,j],mi[:,i],sigma[:,:,i])
        return B

    def forward(self, O):
        T = np.size(O,1)
        alpha = np.zeros((self.N,T))
        B = self.B(O,self.mi,self.sigma)
        alpha[:,0] = self.pi*B[:,0]
        for t in range(T-1):
            alpha[:,t+1] = np.dot(alpha[:,t], self.A)*B[:,t+1]
        
        return alpha
    
    def backward(self, O):
        T = np.size(O,1)
        beta = np.ones((self.N,T))
        B = self.B(O,self.mi,self.sigma)
        for t in range(T-2,-1,-1):
            beta[:,t] = np.dot(self.A, B[:,t+1]*beta[:,t+1]) 
        return beta
    
    def viterbi_algorithm(self, O):
        T = np.size(O,1)
        B = self.B(O,self.mi,self.sigma)
        delta = np.zeros((self.N,T))
        delta[:,0] = self.pi*B[:,0]
        
        psi = np.zeros((self.N, T))
        
        for t in range(1,T):
            aux = np.dot(delta[:,t-1]*np.eye(self.N),self.A)
            delta[:,t] = np.max(aux,0)*B[:,t]
            psi[:,t] = np.argmax(aux,0)
        
        path = np.zeros((1,T), dtype=int)
        path[0,T-1] = np.argmax(delta[:, T-1])
        for t in range(T-2,-1,-1):
            path[0,t] = psi[path[0,t+1], t+1]
        return path
    
    def baum_welch_algorithm(self, O, n_ite):
        T = np.size(O,1)
        numDim = np.size(O,0)    
        pi_ = np.zeros((self.N,1))
        A_ = np.zeros((self.N, self.N))
        mi_ = np.zeros((numDim, self.N))
        sigma_ = np.zeros((numDim,numDim,self.N))
        B = self.B(O,self.mi,self.sigma)
        #log_prob = np.zeros((n_ite,1))

        for ite in range(n_ite):
            #Forward algorithm [N,T]
            alpha = np.zeros((self.N,T))
            alpha = self.forward(O) 
            log_prob = np.log(np.sum(alpha[:,-1]))
            
            #Backward algorithm [N,T]
            beta = np.ones((self.N,T))
            beta = self.backward(O)
            
            # [N,T]
            gamma = alpha*beta
            gamma = gamma/np.sum(gamma,0)
            
            # [T,N,N]
            csi = np.zeros((T-1,self.N,self.N))
            for t in range(T-1):
                for i in range(self.N):
                    for j in range(self.N):
                        csi[t,i,j] = alpha[i,t]*self.A[i,j]*B[j,t+1]*beta[j,t+1]
                csi[t,:,:] = csi[t,:,:]/np.sum(csi[t,:,:])  
            
            # Update Model
            pi_ = gamma[:,0]
            
            num = np.sum(csi,0)
            den = np.sum(gamma[:,0:-1],1)
            A_ = (num.T/den).T

            for j in range(self.N):
                mi_[:,j] = (np.sum(gamma[j,:]*O,1)/np.sum(gamma[j,:]))
                D = (O - np.expand_dims(self.mi[:,j],1))
                sigma_[:,:,j] = ((D@np.diag(gamma[j,:]))@D.T)/np.sum(gamma[j,:])

            self.A = A_
            self.pi = pi_
            self.mi = mi_
            self.sigma = sigma_
        return log_prob
    
