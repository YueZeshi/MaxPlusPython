import numpy as np

class MaxPlus:
    Zero = -np.inf
    One = 0
    Top = np.inf
    def __init__(self,value,shape=None):
        if shape==None:
            if len(np.shape(value))==0 :
               self.value = value*np.ones((1,1))
            elif len(np.shape(value))==1:
                self.value = np.array(value)*np.ones((np.shape(value)[0],1))
            elif len(np.shape(value))==2:
                self.value = np.array(value)
            else:
                return NotImplementedError 
            self.value = np.atleast_2d(value)
            self.shape = np.shape(self.value)
        else:
            self.shape = shape
            self.value = value*np.ones(shape)
    # operator
    def __add__(self,other):
        # broadcast to implement
        if(type(other)!=MaxPlus):
            other = MaxPlus(other,self.shape)
        return MaxPlus(np.max([self.value,other.value],axis = 0))
    def __neg__(self):
        return MaxPlus(-self.value)
    def __sub__(self,other):
        return self+(-other)
    def __matmul__(self,other):
        if type(other)!=MaxPlus:
            other = MaxPlus(other,self.shape)
        if(self.shape[1]==other.shape[0]):
            res = MaxPlus.zeros((self.shape[0],other.shape[1]))
            for i in range(np.shape(res)[0]):
                for j in range(np.shape(res)[1]):
                    for k in range(self.shape[1]):
                        res[i,j] = np.max((res[i,j],self.value[i,k]+other.value[k,j]),axis=0)
            return res
        else:
            raise ValueError(f"matrix size error {np.shape(self.value)},{np.shape(other.value)}")
    def __mul__(self,other):
        if type(other)!=MaxPlus:
            other = MaxPlus(other,self.shape)
        if(self.shape==other.shape):
            res = self.zeros(self.shape)
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    res[i,j] = self.value[i,j]+other.value[i,j]
            return res
        else:
            raise ValueError(f"matrix size error {np.shape(self.value)},{np.shape(other.value)}")
    def __pow__(self,n):
        if n%1==0:
            if n==0:
                return self.eye(self.shape)
            res = self
            for i in range(n-1):
                res=res@self
            return res
        else:
            return NotImplemented
    def __floordiv__(self,other):
        return -(self.T()@(-other)).T()
    def __truediv__(self,other):
        return (self.T()@other.T()).T()
    def __repr__(self):
        return f"{self.value}"
    # comparaison
    def __eq__(self,other):
        if (self.value==other.value).all():
            return True
        else:
            return False
    def __lt__(self,other):
        if (self.value<other.value).all():
            return True
        else:
            return False
    def __gt__(self,other):
        return other<self
    def __le__(self,other):
        if self<other or self==other:
            return True
        return False
    def __ge__(self,other):
        return other<=self
    # slice
    def __getitem__(self,index):
        if isinstance(index,tuple):
            return self.value[index]
        elif isinstance(index,int):
            return self.value[index][index]
        else:
            raise TypeError("Index wrong")
    def __setitem__(self,index,value):
        if isinstance(index,tuple):
            self.value[index[0]][index[1]]=value
        elif isinstance(index,int):
            self.value[index][index]= value
        else:
            raise TypeError("Index error")
    
    # instance method
    def trace(self):
        a,b = self.shape
        l = min(a,b)
        res = -np.inf
        for i in range(l):
            res = max(res,self.value[i,i])
        return res
    def T(self):
        value = self.value.T
        return MaxPlus(value)

    @classmethod
    def eye(cls,size):
        if type(size)==tuple:
            res = cls.zeros(size)
            for i in range(min(size[0],size[1])):
                res[i,i]=MaxPlus.One
            return res
        if type(size)==int:
            res = cls.zeros((size,size))
            for i in range(size):
                res[i,i] = MaxPlus.One
            return res
    
    @classmethod
    def ones(cls,shape):
        return cls.units(shape)
    @classmethod
    def units(cls,shape):
        return cls(np.zeros(shape))
    @classmethod
    def zeros(cls,shape=None):
        if not shape:
            return cls(MaxPlus.Zero)
        return cls(-np.infty*np.ones(shape))
    @classmethod
    def zeros_like(cls,ml):
        return cls.zeros(ml.shape)
    @classmethod
    def top(cls):
        return cls(MaxPlus.Top)
    @classmethod
    def inspan(cls,m,x):# todo
        return True
    @classmethod
    def equalspan(cls,m1,m2):# todo
        return True
    @classmethod
    def star(cls,m):
        if(m.shape[0]!=m.shape[1]):
            raise TypeError("Matrix not square")
        else:
            res = cls.ones(m.shape)
            for i in range(m.shape[0]):
                res = res+m**(i+1)
            return res
    @classmethod
    def plus(cls,m_maxplus):
        maxIter = 1000
        m = m_maxplus
        res0 = cls.zero(m.shape)
        res1 = m
        i=2
        while res0!=res1 and i<maxIter:
            res0 = res1
            res1 =res1+m**i
            i+=1
        if i==1000:
            raise RuntimeError("Not converge")
        return res1
    @classmethod
    def naiveeigenv(cls,m_maxplus):
        if(m_maxplus.shape[0]==m_maxplus.shape[1]):
            res = cls.Zero
            x = m_maxplus
            for i in range(1,m_maxplus.shape[0]+1):
                res=np.max((res, x.trace()/i))
                x = x@m_maxplus
            return res
        else:
            raise ValueError("Matrix not square")
    @classmethod
    def eigenspace(cls,m):
        eigenv = cls.naiveeigenv(m)
        mprime = m*(-eigenv)
        return cls.plus(mprime)[:,0:1],eigenv
    @classmethod
    def astarb(cls,A,b):
        Astar = MaxPlus.star(A)
        return Astar@b
    



class MaxPlusLinearSystem:
    A:MaxPlus
    B:MaxPlus
    C:MaxPlus
    D:MaxPlus
    x0:MaxPlus
    def __init__(self,A,B,C,D = None,x0 = None):
        self.A = A
        self.B = B
        self.C = C
        if not D:
            self.D = MaxPlus.zeros(A)
        if not x0:
            self.x0 = MaxPlus.zeros((A.shape[0],1))
        
    def explicit(self):
        pass

    @classmethod
    def mpsyslin(cls,A,B,C,D=None,x0=None):
        return MaxPlusLinearSystem(A,B,C,D,x0)
    @classmethod
    def explicit(cls,sys):
        pass
    @classmethod
    def simul(cls,S,u):
        pass
