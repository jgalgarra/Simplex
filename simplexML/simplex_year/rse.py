# -*- coding: utf-8 -*-
"""
Created on Wed May  6 09:28:40 2020

@author: yo

Función auxiliar para calcular el RSE

"""
import numpy as np

def calc_rse(valores,mse):
    return(len(valores)*mse/sum((np.mean(valores)-valores)**2))