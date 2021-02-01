# -*- coding: utf-8 -*-
"""
Auxiliar function to compute Root Squared Error
"""
import numpy as np

def calc_rse(valores,prediccion):
    return(sum((valores-prediccion)**2)/sum((valores-np.mean(valores))**2))