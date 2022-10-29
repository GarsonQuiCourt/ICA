import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy.special
import scipy.stats
import random
import re
import time

import warnings
warnings.filterwarnings('ignore')

import librosa
import librosa.display
import IPython.display as ipd
import IPython
import numpy as np
import wave

def shuffle_transform(idx_sort):
    T = np.zeros((len(idx_sort), len(idx_sort)))

    for i in range(len(idx_sort)):
        T[i, idx_sort[i]] = 1
    
    return T

def whitening(X, *, sort_flag=False, fudge=1e-48):
    COV = np.dot(X, X.T) / X.shape[1]
    w, V = np.linalg.eigh(COV)

    T = np.eye(X.shape[0])
    if sort_flag:
        idx_sort = np.argsort(-w)
        T = shuffle_transform(idx_sort)

    W = np.diag(1 / np.sqrt(w + fudge))

    return np.dot(T, np.dot(W, V.T))

def symm_orth(W, fudge=1e-18):
    d, V = np.linalg.eigh(np.dot(W, W.T))
    T = np.dot(V, np.dot(np.diag(1 / np.sqrt(d + fudge)), V.T))
    return np.dot(T, W)

def show(X, s=1, lim=8, ax=None):
    if ax is not None:
        pd.plotting.scatter_matrix(pd.DataFrame(X.T), figsize=(lim * (X.shape[0] / 4),lim * (X.shape[0] / 4)), ax=ax, s=s)
    else:
        pd.plotting.scatter_matrix(pd.DataFrame(X.T), figsize=(lim * (X.shape[0] / 4),lim * (X.shape[0] / 4)), s=s)
        plt.show()

def signal_close(s1, s2):
    s1 /= (np.linalg.norm(s1) + 1e-10)
    s2 /= (np.linalg.norm(s2) + 1e-10)
    return "q0.5 = {}, q99 = {}".format(quantile(abs(abs(s1) - abs(s2)), 0.5), quantile(abs(abs(s1) - abs(s2)), 0.99))


class FastICA:
    def __init__(self, 
                 n = None,
                 G=lambda x: x**3,
                 g=lambda x: 3,
                 max_iter_count=100,
                 sort_flag=True,
                 accuracy=1e-5,
                 fudge=1e-20,):
        '''
            n - количесво независимых компонент, если не будет инициализированна, 
                количество независимых коммпонент будет равно количеству сигналов
            G, g - функция и ее производная для приближенного вычисления гауссиановости
        '''
        # количество независимых компонент
        self.n = n

        # функции для оценки гауссиановости
        self.G = G
        self.g = g

        # количество итераций
        self.max_iter_count = max_iter_count

        # флаг сортировки
        self.sort_flag = sort_flag

        # точность
        self.accuracy = accuracy

        # Используется для избежания перегрузки чисел
        self.fudge = fudge

        # Матрица преобразоваия
        self.Transform = 1


    def fit(self, 
            signal,
            W=None,
            show_flag=False,
            duration_show=False,
            iter_to_show=range(0, 100, 10)):
        
        if duration_show: 
            t_start = time.time()
            Time = [0]

        signal_sample_num = signal.shape[1]
        signal_comp_num = signal.shape[0]

        # Правило для максимизации негауссиановости
        def rule(w, X):
            return X * self.G(np.dot(w.T, X)) - w.reshape(-1, 1) * np.ones((1, X.shape[1])) * self.g(np.dot(w.T, X))

        
        if self.n is None:
            self.n = signal_comp_num

        # Запоминае смещения для дальнейших преобразований
        self.bias = np.mean(signal, axis=-1).reshape(-1, 1)
        X = signal - self.bias

        self.Transform = whitening(X, fudge=self.fudge, sort_flag=self.sort_flag)

        X = np.dot(self.Transform, X)

        if show_flag:
            to_show = []
            to_show.append((0, X.copy()))
        
        # Инициализация матрицы весов
        if W is None:
            W = [np.random.random_sample(size=signal_comp_num) for _ in range(self.n)]
            for w in W:
                w /= np.linalg.norm(w)
            W = np.vstack(W)
            W = symm_orth(W)

        if duration_show: 
            Time.append(time.time() - t_start)

        for i in range(1, self.max_iter_count + 1):
            W_prev = W.copy()

            for k in range(self.n):
                W[k, :] = np.mean(rule(W[k, :], X), axis=-1)
            
            W /= np.linalg.norm(W) / self.n
            W = symm_orth(W)

            # Странное правило для сходимости, но вообще я хочу смотреть на 
            # коэффициент эксцесса и ставить ограничение по количеству итераций самостоятельно
            if np.linalg.norm(np.abs(W) - np.abs(W_prev)) <= self.accuracy:
                break
            
            if show_flag and i in iter_to_show:
                to_show.append((i, np.dot(W, X)))
            
            if duration_show: 
                Time.append(time.time() - t_start)
        
        self.Transform = np.dot(W, self.Transform)
        
        if show_flag: return to_show
        if duration_show: return Time

    
    def transform(self, signal):
        return np.dot(self.Transform, signal)

class SS_NonstationaryVar:
    def __init__(self, 
                 n=None,
                 max_iter_count=100,
                 sort_flag=False,
                 tol=1e-5,
                 fudge=1e-20,
                 dt=50,
                 alpha=0.95):
        # количество независимых компонент
        self.n = n

        # количество итераций
        self.max_iter_count = max_iter_count

        # флаг сортировки
        self.sort_flag = sort_flag

        # точность
        self.tol = tol

        # Используется для избежания перегрузки чисел
        self.fudge = fudge

        # Матрица преобразования
        self.Transform = 1

        # Промежуток для оценки локальной лиспесии
        self.dt = dt

        # Константа для геометрического ряда сходимости
        self.alpha = alpha


    def fit(self, 
            signal,
            W=None,
            show_flag=False,
            iter_to_show=range(0, 100, 10)):
        
        signal_sample_num = signal.shape[1]
        signal_comp_num = signal.shape[0]

        def rule(W, X, dt):
            dW = 0
            p = np.cumsum(np.dot(W.T, X) ** 2, axis=1)
            for t in range(dt, X.shape[1], dt):
                dW += np.dot(np.diag(((p[:, min(t + dt, p.shape[1] - 1)] - p[:, max(t - dt - 1, 0)]) / (min(t + dt, p.shape[1] - 1) - max(t - dt - 1, 0)) + self.fudge) ** -1), 
                             np.dot(W, np.dot(X[:, t].reshape(-1,1), X[:, t].reshape(1, -1))))
            return dW

        
        if self.n is None:
            self.n = signal_comp_num

        self.bias = np.mean(signal, axis=-1).reshape(-1, 1)
        X = signal - self.bias

        self.Transform = whitening(X, fudge=self.fudge, sort_flag=self.sort_flag)

        X = np.dot(self.Transform, X)
        
        if W is None:
            W = np.random.random_sample(size=(self.n, signal_comp_num))
            W /= np.linalg.norm(W) / self.n
            W = symm_orth(W)
        if show_flag:
            print(W)

        a = 1
        for i in range(1, self.max_iter_count + 1):
            W_prev = W.copy()

            W = W + a * rule(W, X, self.dt)
            a *= self.alpha
            
            W /= np.linalg.norm(W) / self.n
            W = symm_orth(W)

            if show_flag:
                print(i, a)
                print(W)
                print()
        
        self.Transform = np.dot(W, self.Transform)
        return np.dot(W, X)

    
    def transform(self, signal):
        return np.dot(self.Transform, signal - self.bias)
