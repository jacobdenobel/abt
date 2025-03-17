import os
from dataclasses import dataclass

import numpy as np

from pymatreader import read_mat

ROOT = os.path.realpath(os.path.dirname(__file__))
DEFAULT_T = os.path.join(ROOT, "Fidelity120 HC3A MS All Morphologies 18us CF.mat")
IDET_NO_CS = os.path.join(ROOT, "idet_no_cs.npy")

@dataclass
class MatlabData:
    M: np.ndarray
    T: np.ndarray
    I_det: np.ndarray
    greenwood_f: np.ndarray
    mat: object = None
    pw: float = 18e-6
    
    @staticmethod
    def read(
        path: str = DEFAULT_T, 
        m: int = 0, 
        compute_tm: bool = True
    ) -> "MatlabData":
        
        mat = read_mat(path)
        TIa = mat["Df120"]["TIa"][m] * 1e-3
        TIb = mat["Df120"]["TIb"][m] * 1e-3
        I_det = TIa + TIb
        I_det = np.nan_to_num(I_det, nan=np.nanmax(I_det, axis=0))
        
        if not compute_tm:
            M = mat['Df120']['M'][m] * 1e-3
            T = mat['Df120']['T'][m] * 1e-3
        else:
            T, M = matlab_TM_thresholds(
                np.load(IDET_NO_CS),
                np.flipud(mat["Df120"]["Ln"][0])   
            )
        
        return MatlabData(
            M=M,
            T=T,
            I_det = MatlabData.transform_to_virtual(I_det),
            greenwood_f = np.flip(mat['Df120']['Fn'][m] * 1e3),
            mat=mat
        )
        
        
    @staticmethod
    def transform_to_virtual(i_det):
        reshaped = i_det[:, :i_det.shape[1], :].reshape(-1, i_det.shape[2]).T
        return np.flip(reshaped, axis=0)
        # reshaped = i_det[:, :i_det.shape[1] - 1, :].reshape(-1, i_det.shape[2]).T
        # return np.flip(np.c_[reshaped, i_det[-1, -1, :]], axis=0)
    
    def spatial_factor(self, fiber_idx, i_det):
        return np.nanmin(i_det, axis=0) / i_det[fiber_idx, :]

    def sigma(self, fiber_idx, i_det):
        return i_det[fiber_idx, :] * 0.06
    
def TPloudness(TP,VL):
    # Bepaal luidheidsgroeicurve uit drempelprofiel
    # Let op: gaten in het drempelprofiel komen aan het eind van output-array I (en dus ook L) te staan!
    dL=0.5*abs(VL[1:]-VL[:-1])       # Helft van verschillen in afstand langs OC voor alle naburige vezels
    Wf=np.zeros(VL.shape)               # Array voor OC-breedte per vezel
    Wf[:-1] = Wf[:-1] + dL            # Helft van afstand naar volgende vezels
    Wf[1:]   = Wf[1:] + dL          # Plus de helft van de afstand naar de vorige vezels
    Wf[[0,-1]] = 2*Wf[[0, -1]]              # Eerste en laatste vezel worden gecorrigeerd voor het feit dat ze geen vorige respectievelijk volgende vezel hebben
    I, n = matlab_sort(TP)       # Drempelcurve wordt gesorteerd
    I = np.asarray(I)
    n = np.asarray(n)
    Wf_sort = Wf[n].ravel()                     # Excitatiebreedte per vezel wordt op dezelfde volgorde gesorteerd
    L=np.zeros(I.shape)                 # Array voor luidheid (in mm langs OC)
    L[0]= Wf_sort[0]                         # Excitatiebreedte laagstdrempelige vezel is Wf(1)
    for i in np.arange(1,len(I)):       # Elke volgende vezel voegt Wf(i) aan de excitatiebreedte toe
        L[i] = L[i-1]+Wf_sort[i]        #

    return I, L

def interpolindex(val, A):
    # Interpoleert een index n behorende bij een waarde val in een monotoon stijgend/dalend array A
    # Gebruikt lineaire interpolatie

    NA=len(A)
    if A[0] < A[-1]:
        k = 1
        dk = 1
    else:
        k=NA
        dk = -1
    while (A[k]<val) and (k+dk>=1) and (k+dk<=NA):
        k = k + dk
    if (A[k]>=val) and (k-dk>=1) and (k-dk<=NA):
        n1 = k-dk
        n2 = k 
        index=n1+1 + dk*(val-A[n1])/(A[n2]-A[n1]) # +1 for matlab indexing
    else:
        index=k

    return index

def interpolval(index,A):
    if not np.isnan(index):
        c = index-np.floor(index)
        n = int(np.floor(index))
        if n<len(A):
            if n>=1:
                val = (1-c)*A[n-1] + c*A[n]
            else:
                val = A[0]
        else:
            val = A[-1]
    else:
        val = np.nan

    return val

def DetermineThresholdLevel(TP,VL,Level):
    # TP    : drempelprofiel
    # VL    : posities van vezels langs basilair membraan (in mm)
    # Level : luidheidsniveau waarvoor stroomniveau bepaald moet worden (in mm)

    I,L =TPloudness(TP,VL)    # Bepaal luidheidsgroeicurve (I,L) uit drempelprofiel

    k=interpolindex(Level,L)  # Interpoleer de index van Level in L
    Itl=interpolval(k,I);     # Interpoleer de waarde van I op index k

    return Itl

def matlab_sort(array):
    if len(array.shape) > 1:
        array = np.squeeze(array)
    indices = [] 
    values = [] 
    remaining_array = array[~np.isnan(array)]
    while len(remaining_array) > 0 :
        min_val = np.nanmin(remaining_array)
        min_index = np.where(array == min_val)[0].tolist() # already sorts it ascendingly
        if len(min_index) > 1:
            indices.extend(min_index)
            values.extend(array[min_index])
        else:
            indices.append(min_index[0])
            values.append(array[min_index[0]])
        remaining_array = np.delete(remaining_array, np.where(remaining_array == min_val))

    return values, indices

def matlab_TM_thresholds(I_det, Ln, T_mm = 0.5, M_mm = 3.5):
    _, num_electrodes = I_det.shape
    T = np.zeros(num_electrodes)
    M = np.zeros(num_electrodes)
    

    for e in np.arange(num_electrodes):
        T[e] = DetermineThresholdLevel(np.expand_dims(I_det[:,e], axis=1), np.expand_dims(Ln, axis=1), T_mm) 
        M[e] = DetermineThresholdLevel(np.expand_dims(I_det[:,e], axis=1), np.expand_dims(Ln, axis=1), M_mm) 
    return T, M