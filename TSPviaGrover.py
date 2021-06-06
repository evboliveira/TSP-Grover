import numpy as np
import qutip as qt
import qiskit as qk
from itertools import permutations 
import matplotlib.pyplot as plt
#import igraph as ig
import seaborn as sns
import matplotlib.ticker as tck
import sys


class TSP_hamiltonian_paths_graph:
    '''This class presents methods for the formulation of the TSP problem in terms of hamiltonian cicles of
    a set of random cities
    '''
        
    def __init__(self, num_cities, random = True):
        self.num_cities = num_cities
        self.random = random
        
        if self.random:            
                self.cities=self.set_aleatory_cities(self.num_cities)
    
    
    def set_aleatory_cities(self,N,area=(10,10)):
        '''Generates a set os N randomly distribuited on an 10x10 square area'''
        N=self.num_cities
        cities = np.random.rand(N, 2)
        cities[:,0] *= area[0]
        cities[:,1] *= area[1]
        return cities
    
    
    def matrixD(self, rescaled=True):
        '''Return the distance matrix of the problem, that each D_{ij} matrix element represents
        the distance beteween the ith and the jth-city. If "rescaled" is set "True", then it returns 
        the "rescaled matrixD", i.e., the range distances varying between [0,2 pi]
        '''
        N=self.num_cities
        cities = self.cities
        matrixD=np.zeros((N,N))
        
        for i in range(0,N):
            for j in range(0,N):
                p1=cities[i]
                p2=cities[j]
                if type(p1) == np.ndarray and type(p2) == np.ndarray:
                    matrixD[i,j] = np.sqrt(np.sum(np.power(p1 - p2, 2)))
                else:
                    raise TypeError('p1 and p2 must be np.ndarray')
        if rescaled: 
            D = matrixD
            Dmin = np.amin(D[np.nonzero(D)])
            Dmax = np.amax(D)
            delta = Dmax-Dmin
            
            for i in range(len(D)):
                for j in range(len(D)):
                    if i!=j :
                        D[i][j] = ((D[i][j]-Dmin)/(N*delta))*2*np.pi
            matrixD = D
        return matrixD    
    
    
    def hamiltonian_paths(self, rescaled=True):
        '''Return a list of all the Hamiltonian paths' distances. The list order is set by the
        permutation of the cities. If "rescaled" is set "True", then it returns 
        the "rescaled Hamiltonian paths", i.e., the range distances varying between [0,2 pi]
        '''
        
        N = self.num_cities
        num_paths = np.math.factorial(N-1)
        
        if not rescaled:
            D = self.matrixD(False)
        else:
            D = self.matrixD()
        
        H_paths = np.zeros(num_paths)
        l = list(permutations([i for i in range(1,N)]))
        for n in range(num_paths):
            H_paths[n] = D[0][l[n][0]] + D[l[n][-1]][0]
            for i in range(N-2):
                H_paths[n] += D[l[n][i]][l[n][i+1]]
        return H_paths
    
    
    def hamiltonian_paths_histogram(self, rescaled):
        '''Return a histogram of the Hamiltonian paths distribuition acording to their distances
        '''
        
        N=self.num_cities
        num_paths = np.math.factorial(N-1)
        sns.distplot(self.hamiltonian_paths(rescaled)[0], 
                     hist=True, kde=True,
                     bins=int(100), 
                     color = 'darkblue',
                     hist_kws={'edgecolor':'black'},
                     kde_kws={'linewidth': 2}
                    )
        plt.title("N="+str(self.num_cities)+" cidades")
        plt.grid(True)
        plt.xlim(0,2*np.pi)
        plt.ylim(0,1.2)
        plt.vlines(np.pi, 0, 1, colors='r')
        plt.show()
        
    def optimum_path(self, rescale=True):
        '''Returns the Hamiltonian paths and their respective distances sorted from the
        shortest to the longest
        '''
        
        N = self.num_cities
        num_paths = np.math.factorial(N-1)
        H_paths = self.hamiltonian_paths(rescale)
        l = list(permutations([i for i in range(1,N)]))
        best = int(np.ceil(num_paths))
        optimum=H_paths
        optimum.sort()
        sort = []
        for i in range(num_paths):
            index = list(H_paths).index(optimum[i])
            sort.append( [ H_paths[i] , (0,)+l[index] ] )
        return sort
    
    def conditonal_operator(self):
        '''Creates the conditional operator "P" that acts on a product state of two cities 
        provinding the distance between them in a form of a relative phase that vary from 0 to 2 pi
        '''
        N = self.num_cities
        D = self.matrixD(N)
        cond_P=np.array([])

        D_min=np.amin(D)
        D_max=np.amax(D)

        for i in range (0,num_cities):
            cond_P=np.append( cond_P ,  np.exp( 1j*((D[i]-D_min)/(D_max-D_min))*2*np.pi) )        
        cond_P=np.diag(cond_P)

        return(cond_P)


class quantum_operators_hamiltonian_cicles_via_grover:
    '''
    Class for the creation of the quantum operators related to the TSP
    problem for a number o N cities
    '''
    
    def __init__(self, num_cities):
        self.num_cities = num_cities
        
    def number_of_qbits(self):
        '''set the minimum number of qbits to represent a total of N states'''
        N = self.num_cities
        return int(np.ceil(np.log2(N))) 
    
    
    def city_projector(self, n='index'):
        '''Generates the projector of the vector states that represents the nth-city
        on a list of a total number of N cities
        '''
        N = self.num_cities
        if (n>=N):
            raise TypeError("The operator index n must be less than N='number of cities'")

        else:    
            n_qbits = self.number_of_qbits()
            ket=qt.ket(np.binary_repr(n, width=n_qbits))
            return ket*ket.dag()
        
    
    def path_projector(self, n='index'):
        '''This function returns a projector of the state vector that represents
        the nth-hamiltonian_path on a total of (N-1)! hamiltonian paths
        '''
        N = self.num_cities
        num_paths=np.math.factorial(N-1) #(N-1)! possible hamiltonian paths

        if n>=num_paths:
            raise TypeError("The index n must be less than the total number of paths")

        else:
            l=list(permutations([i for i in range(1,N)]))
            Ulist=[self.city_projector(0)]+[self.city_projector(i) for i in l[n]]
            T=qt.tensor(Ulist)

            return T
        
    def hamiltonian_cicles_oracle_projector(self):
        '''This function returns the projector of the state vector the represents
        a supoerposition states of all the (N-1)! hamiltonian paths'''
        N = self.num_cities
        num_paths = np.math.factorial(N-1) #(N-1)! possible hamiltonian paths
        n_qbits = self.number_of_qbits()
        
        for i in range(0,num_paths):
            if (i==0):
                T = self.path_projector(i)
            else:
                T += self.path_projector(i)
        #T = (T/np.sqrt(num_paths))
        Id = qt.qeye(N*n_qbits*[2])
        return (Id-2*T)
    
    def grover_operator(self, Oracle, n='number of aplications'):
        '''Returns the Grover operator with a specified Oracle operator 
        and number of aplications'''
        dim = Oracle.dims[0]
        n_qbits = len(dim)        
        ket0ket0 = qt.tensor([qt.basis(2,0)]*n_qbits)
        ket0ket0 = ket0ket0*ket0ket0.dag()  
        H = qt.qip.operations.hadamard_transform(n_qbits)
        Id = qt.qeye(dim)

        Grover = (2*H*ket0ket0*H-Id)*Oracle
        return Grover**n

