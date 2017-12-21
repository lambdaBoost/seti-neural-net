# -*- coding: utf-8 -*-
"""
Created on Sun May 15 17:05:30 2016

@author: Alex
"""
#SCRIPT TO GENERATE SIMULATED DETECTIONS. THE OUTPUTS WILL NEED FLATTENING TO A LIST PRIOR TO FEEDING THROUGH THE NN

import random
import itertools
import numpy as np

#function to flatten list of lists to single list
def flatten(lst):
    if lst:
        car,*cdr=lst
        if isinstance(car,(list,tuple)):
            if cdr: return flatten(car) + flatten(cdr)
            return flatten(car)
        if cdr: return [car] + flatten(cdr)
        return [car]
        
        
#SIMULTED NARROWBAND LASER SIGNAL - 1px width - no fade
def generate_narrowband_1():
    a=np.array([[random.uniform(0,0.015) for _ in range(0, 20)],
    [random.uniform(0,0.015) for _ in range(0, 20)],
    [random.uniform(0,0.015) for _ in range(0, 20)],
    [random.uniform(0,0.015) for _ in range(0, 20)],
    [random.uniform(0,0.015) for _ in range(0, 20)],
    flatten([[random.uniform(0,0.02) for _ in range(0, 9)],[random.uniform(0,0.11)],[random.uniform(0,0.02) for _ in range(10, 20)]]),
    flatten([[random.uniform(0.019,0.045) for _ in range(0, 9)],[random.uniform(0.11,0.25)],[random.uniform(0.019,0.045) for _ in range(10, 20)]]),
    flatten([[random.uniform(0.06,0.085) for _ in range(0, 9)],[random.uniform(0.33,0.47)],[random.uniform(0.06,0.085) for _ in range(10, 20)]]),
    flatten([[random.uniform(0.1,0.142) for _ in range(0, 9)],[random.uniform(0.56,0.79)],[random.uniform(0.1,0.142) for _ in range(10, 20)]]),
    flatten([[random.uniform(0.15,0.18) for _ in range(0, 9)],[random.uniform(0.83,1.0)],[random.uniform(0.15,0.18) for _ in range(10, 20)]]),
    flatten([[random.uniform(0.15,0.18) for _ in range(0, 9)],[random.uniform(0.83,1.0)],[random.uniform(0.15,0.18) for _ in range(10, 20)]]),
    flatten([[random.uniform(0.1,0.142) for _ in range(0, 9)],[random.uniform(0.56,0.79)],[random.uniform(0.1,0.142) for _ in range(10, 20)]]),
    flatten([[random.uniform(0.06,0.085) for _ in range(0, 9)],[random.uniform(0.33,0.47)],[random.uniform(0.06,0.085) for _ in range(10, 20)]]),
    flatten([[random.uniform(0.019,0.045) for _ in range(0, 9)],[random.uniform(0.11,0.25)],[random.uniform(0.019,0.045) for _ in range(10, 20)]]),
    flatten([[random.uniform(0,0.02) for _ in range(0, 9)],[random.uniform(0,0.11)],[random.uniform(0,0.02) for _ in range(10, 20)]]),
    [random.uniform(0,0.015) for _ in range(0, 20)],
    [random.uniform(0,0.015) for _ in range(0, 20)],
    [random.uniform(0,0.015) for _ in range(0, 20)],
    [random.uniform(0,0.015) for _ in range(0, 20)],
    [random.uniform(0,0.015) for _ in range(0, 20)]])
    return(a)


#SIMULTED NARROWBAND LASER SIGNAL - 1px width - with fade

def generate_narrowband_2():
    a=np.array([[random.uniform(0,0.015) for _ in range(0, 20)],
    [random.uniform(0,0.015) for _ in range(0, 20)],
    [random.uniform(0,0.015) for _ in range(0, 20)],
    [random.uniform(0,0.015) for _ in range(0, 20)],
    [random.uniform(0,0.015) for _ in range(0, 20)],
    flatten([[random.uniform(0,0.02) for _ in range(0, 8)],[random.uniform(0,0.05)],[random.uniform(0,0.11)],[random.uniform(0,0.05)],[random.uniform(0,0.02) for _ in range(11, 20)]]),
    flatten([[random.uniform(0.019,0.045) for _ in range(0, 8)],[random.uniform(0.05,0.125)],[random.uniform(0.11,0.25)],[random.uniform(0.05,0.125)],[random.uniform(0.019,0.045) for _ in range(11, 20)]]),
    flatten([[random.uniform(0.06,0.085) for _ in range(0, 8)],[random.uniform(0.15,0.23)],[random.uniform(0.33,0.47)],[random.uniform(0.15,0.23)],[random.uniform(0.06,0.085) for _ in range(11, 20)]]),
    flatten([[random.uniform(0.1,0.142) for _ in range(0, 8)],[random.uniform(0.28,0.4)],[random.uniform(0.56,0.79)],[random.uniform(0.28,0.4)],[random.uniform(0.1,0.142) for _ in range(11, 20)]]),
    flatten([[random.uniform(0.15,0.18) for _ in range(0, 8)],[random.uniform(0.41,0.5)],[random.uniform(0.83,1.0)],[random.uniform(0.41,0.5)],[random.uniform(0.15,0.18) for _ in range(11, 20)]]),
    flatten([[random.uniform(0.15,0.18) for _ in range(0, 8)],[random.uniform(0.41,0.5)],[random.uniform(0.83,1.0)],[random.uniform(0.41,0.5)],[random.uniform(0.15,0.18) for _ in range(11, 20)]]),
    flatten([[random.uniform(0.1,0.142) for _ in range(0, 8)],[random.uniform(0.28,0.4)],[random.uniform(0.56,0.79)],[random.uniform(0.28,0.4)],[random.uniform(0.1,0.142) for _ in range(11, 20)]]),
    flatten([[random.uniform(0.06,0.085) for _ in range(0, 8)],[random.uniform(0.15,0.23)],[random.uniform(0.33,0.47)],[random.uniform(0.15,0.23)],[random.uniform(0.06,0.085) for _ in range(11, 20)]]),
    flatten([[random.uniform(0.019,0.045) for _ in range(0, 8)],[random.uniform(0.05,0.125)],[random.uniform(0.11,0.25)],[random.uniform(0.05,0.125)],[random.uniform(0.019,0.045) for _ in range(11, 20)]]),
    flatten([[random.uniform(0,0.02) for _ in range(0, 8)],[random.uniform(0,0.05)],[random.uniform(0,0.11)],[random.uniform(0,0.05)],[random.uniform(0,0.02) for _ in range(11, 20)]]),
    [random.uniform(0,0.015) for _ in range(0, 20)],
    [random.uniform(0,0.015) for _ in range(0, 20)],
    [random.uniform(0,0.015) for _ in range(0, 20)],
    [random.uniform(0,0.015) for _ in range(0, 20)],
    [random.uniform(0,0.015) for _ in range(0, 20)]])
    return(a)
    
#SIMULATED NARROWBAND SIGNAL WITH NO FADE AND COSMIC RAY STRIKE    
def generate_narrowband_1CR():
    #generate a signal and dim it (dimmed relative to CR strike)
    a=generate_narrowband_1()
    a=a*(random.uniform(0.3,0.9))
    #generate random coordinates for a ray strike:
    rowid=random.randint(0,20)
    colid=random.randint(0,20)
    
    #replace the rnadomly selected pixel with a value of 1
    a[rowid][colid]=1
    return(a)
    
#AS ABVE BUT INCLUDES FADE IN THE SIGNAL
def generate_narrowband_2CR():
        #generate a signal and dim it (dimmed relative to CR strike)
    a=generate_narrowband_2()
    a=a*(random.uniform(0.3,0.9))
    #generate random coordinates for a ray strike:
    rowid=random.randint(0,20)
    colid=random.randint(0,20)
    
    #replace the rnadomly selected pixel with a value of 1
    a[rowid][colid]=1
    return(a)