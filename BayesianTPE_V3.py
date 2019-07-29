from __future__ import print_function
from __future__ import division
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.neighbors import KernelDensity
import numpy as np
from scipy.stats import norm



def _hashable(x):
    return tuple(map(float,x))


def unique_rows(a):
    if a.size == 0:
        return np.empty((0,))

    order = np.lexsort(a.T)
    reorder = np.argsort(order)

    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1)

    return ui[reorder]

def ensure_rng(random_state=None):

    if random_state is None:
        random_state = np.random.RandomState()
    elif isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    else:
        assert isinstance(random_state, np.random.RandomState)
    return random_state


class UtilityFunction(object):

    def __init__(self,kind,kappa,xi):

        self.kappa=kappa

        self.xi=xi

        if kind not in ['ucb','ei','poi','tpe']:
            err="Please choose one of ucb,ei,poi"
            raise NotImplementedError(err)
        else:
            self.kind=kind


    def utility(self,x,gp,y_max,xt,yt):
        if self.kind=='ucb':
            return self._ucb(x,gp,self.kappa)
        if self.kind=='ei':
            return self._ei(x,gp,y_max,self.xi)
        if self.kind=='poi':
            return self._poi(x,gp,y_max,self.xi)
        if self.kind=='tpe':
            return self.tpe(x,xt,yt)

    @staticmethod
    def _ucb(x,gp,kappa):
        mean,std=gp.predict(x,return_std=True)
        return mean+kappa*std


    @staticmethod
    def _ei(x,gp,y_max,xi):
        mean,std=gp.predict(x,return_std=True)
        z=(mean-y_max-xi)/std
        return (mean-y_max-xi)*norm.cdf(z)+std*norm.pdf(z)

    @staticmethod
    def _poi(x,gp,y_max,xi):
        mean,std=gp.predict(x,return_std=True)
        z=(mean-y_max-xi)/std
        return norm.cdf(z)

    @staticmethod
    def tpe(x_trail,x,y):

        best_ratio = 0.2
        n_best = int(len(y) * best_ratio)
        y_sorted_index = np.argsort(y)

        k=0
        x_sorted=np.zeros([len(x),len(x[0])])
        for i in y_sorted_index:
            for j in range(len(x[0])):
                x_sorted[k][j]=x[i][j]
            k+=1

        x_best = x_sorted[len(y) - n_best:]
        x_worst = x_sorted[:len(y) - n_best]

        kd_b=KernelDensity(bandwidth=0.1)
        kd_w=KernelDensity(bandwidth=0.1)
        kd_best=kd_b.fit(x_best)
        kd_worst=kd_w.fit(x_worst)

        best_proba=kd_best.score_samples(x_trail)
        worst_proba=kd_worst.score_samples(x_trail)

        ei=np.abs(worst_proba)/np.abs(best_proba)

        return ei

class TargetSpace:

    def __init__(self,f,bounds,random_state=None):


        self.random_state =ensure_rng(random_state)

        self.f=f
        self.keys=list(bounds.keys())
        self.bounds=np.array(list(bounds.values()),dtype=np.float)
        self.dim=len(self.keys)

        self._length = 0 # of observations


        self._Xarr=None
        self._Yarr=None

        # views of preallocated data
        self._Xview=None
        self._Yview = None
        self._cache={}

    @property
    def getX(self):
        return self._Xview

    @property
    def getY(self):
        return self._Yview

    def __contains__(self, x):
        return _hashable(x) in self._cache

    def __len__(self):
        return self._length

    def _dict_to_points(self,points_dict):

        param_tup_lens=[]

        for key in self.keys:
            param_tup_lens.append(len(list(points_dict[key])))


        if all([e==param_tup_lens[0] for e in param_tup_lens]):
            pass
        else:
            raise ValueError('The same number of initialization points must be entered for every parameter')

        all_points=[]
        for key in self.keys:
            all_points.append(points_dict[key])


        points=list(map(list,zip(*all_points)))
        return points


    def observe_point(self,x):
        '''finding y=f(x)'''

        x=np.asarray(x).ravel()
        assert x.size==self.dim,"x must have the same dimension"

        if x in self:
            '''lookup for the cache'''
            y=self._cache[_hashable(x)]
        else:
            '''find the target function'''
            params=dict(zip(self.keys,x))
            y=self.f(**params)
            self.add_observation(x,y)
        return y




    def add_observation(self,x,y):

        if self._length>=self._n_alloc_rows:
            self.allocate((self._length+1)*2)

        x=np.asarray(x).ravel()

        self._cache[_hashable(x)]=y

        self._Xarr[self._length] = x
        self._Yarr[self._length] = y

        self._length+=1

        self._Xview=self._Xarr[:self._length]
        self._Yview=self._Yarr[:self._length]



    def allocate(self,num,fast=True):

        if num <=self._n_alloc_rows:
            raise ValueError('num must be larger than current array length')

        self._assert_internal_invariants()

        _Xnew=np.empty((num,self.bounds.shape[0]))
        _Ynew=np.empty(num)


        if self._Xarr is not None:
            _Xnew[:self._length]=self._Xarr[:self._length]
            _Ynew[:self._length]=self._Yarr[:self._length]

        self._Xarr=_Xnew
        self._Yarr=_Ynew

        self._Xview=self._Xarr[:self._length]
        self._Yview=self._Yarr[:self._length]

    @property
    def _n_alloc_rows(self):
        return 0 if self._Xarr is None else self._Xarr.shape[0]

    def random_points(self,num):
        '''Creates random points within the bounds of a space variable'''

        data=np.empty((num,self.dim))

        for col,(lower,upper) in enumerate(self.bounds):
             data.T[col]=self.random_state.uniform(lower,upper,size=num)
        return data

    def max_point(self):
        '''Return current max points that best maximizes the target function'''

        return {'max_val':self.getY.max(),
                'max_params': dict(zip(self.keys,
                                       self.getX[self.getY.argmax()]))}

    def _assert_internal_invariants(self, fast=True):
        """
        Run internal consistency checks to ensure that data structure
        assumptions have not been violated.
        """
        if self._Xarr is None:
            assert self._Yarr is None
            assert self._Xview is None
            assert self._Yview is None
        else:
            assert self._Yarr is not None
            assert self._Xview is not None
            assert self._Yview is not None
            assert len(self._Xview) == self._length
            assert len(self._Yview) == self._length
            assert len(self._Xarr) == len(self._Yarr)

            if not fast:
                # run slower checks
                assert np.all(unique_rows(self.X))


def acq_max(ac,gp,y_max,xt,yt,bounds,random_state,n_warmup=100000,n_iter=250):
    '''function to find the maximum of the acquisition function'''
    x_tries=random_state.uniform(bounds[:,0],bounds[:,1],
                                 size=(n_warmup,bounds.shape[0]))
    ys=ac(x_tries,gp=gp,y_max=y_max,xt=xt,yt=yt)
    x_max=x_tries[ys.argmax()]
    max_acq=ys.max()

    x_seeds=random_state.uniform(bounds[:,0],bounds[:,1],
                                 size=(n_iter,bounds.shape[0]))

    for x_try in x_seeds:
        res=minimize(lambda x:-ac(x.reshape(1,-1),gp=gp,y_max=y_max,xt=xt,yt=yt),
                     x_try.reshape(1,-1),
                     bounds=bounds,
                     method="L-BFGS-B")


        #see if success
        if not res.success:
            continue

        #store if better than the previous minimum/maximum
        if max_acq is None or -res.fun[0]>=max_acq:
            x_max=res.x
            max_acq=-res.fun[0]

    return np.clip(x_max,bounds[:,0],bounds[:,1])



class Optimizer():
    def __init__(self,f,bounds,random_state=None,verbose=1):
        self.bounds=bounds
        self.random_state=ensure_rng(random_state)
        self.gp=GaussianProcessRegressor(kernel=Matern(nu=2.5),
                                         n_restarts_optimizer=25,
                                         random_state=self.random_state)
        self.space=TargetSpace(f,bounds,self.random_state)
        self.initialized = False
        self.i = 0# used for iterations
        self.util = None #utility function

        self._acqkw = {'n_warmup': 100000, 'n_iter': 250}
        self.init_points = []
        self.x_init = []
        self.y_init = []

        self.res={} #output dictionary
        self.res['max']={'max_val':None,
                         'max_params': None}
        self.res['all']={'values':[],'params':[]}
        self.verbose = verbose

    def init(self,init_points):
        #init_points is number of randomly sampled points to probe
        rand_points=self.space.random_points(init_points)
        self.init_points.extend(rand_points)


        for x in self.init_points:
            y=self.space.observe_point(x)

        if self.x_init:
            x_init=np.vstack(self.x_init)
            y_init=np.hstack(self.y_init)
            for x,y in zip(x_init,y_init):
                self.space.add_observation(x,y)


        self.initialized=True


    def explore(self,points_dict,eager=False):

        if eager:
            points=self.space._dict_to_points(points_dict)
            for x in points:
                self.space.observe_point(x)
        else:
            points=self.space._dict_to_points(points_dict)
            self.init_points=points


    def maximize(self,init_points=5,n_iter=25,acq='ei',kappa=2.576,xi=0.0,**gp_params):
        '''main optimization method'''
        self.util=UtilityFunction(kind=acq,kappa=kappa,xi=xi)

        if not self.initialized:
            self.init(init_points)

        y_max=self.space.getY.max()

        self.gp.set_params(**gp_params)

        self.gp.fit(self.space.getX,self.space.getY)

        x_max=acq_max(ac=self.util.utility,
                      gp=self.gp,
                      y_max=y_max,
                      yt=self.space.getY,
                      xt=self.space.getX,
                      bounds=self.space.bounds,
                      random_state=self.random_state,
                      **self._acqkw)

        for i in range(n_iter):
            while x_max in self.space:
                x_max=self.space.random_points(1)[0]

            y=self.space.observe_point(x_max)

            #update the gp
            self.gp.fit(self.space.getX,self.space.getY)

            #update the best params seen so far
            self.res['max']=self.space.max_point()
            self.res['all']['values'].append(y)
            self.res['all']['params'].append(dict(zip(self.space.keys,x_max)))

            if self.space.getY [ -1] > y_max:
                y_max=self.space.getY [-1]


            #maximize the acquisition function to find the next probing point
            x_max=acq_max(ac=self.util.utility,
                          gp=self.gp,
                          y_max=y_max,
                          yt=self.space.getY,
                          xt=self.space.getX,
                          bounds=self.space.bounds,
                          random_state=self.random_state,
                          **self._acqkw)

            self.i +=1