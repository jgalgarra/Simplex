# -*- coding: utf-8 -*-
"""
Created on Wed May  6 09:28:40 2020

@author: yo

Función auxiliar para calcular el RSE

"""
import numpy as np

def calc_rse(valores,prediccion):
    return(sum(valores-prediccion)**2/sum((valores-np.mean(valores))**2))