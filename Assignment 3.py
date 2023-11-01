#!/usr/bin/env python
# coding: utf-8

# # Data Analysis Homework 3
# ### Due date: Tuesday, October 25th 2024, 2 PM

# In[194]:


from __future__ import division
from IPython.display import HTML
from IPython.display import display
from scipy.optimize import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ## Question 1: Linear Regression, Curvature Matrix
# 
# Consider the data listed below,
# \begin{equation}
# \begin{array}{lcccccc}
# \hline
# {\rm frequency~(Hz)} &10&20&30&40&50&60\\
# {\rm voltage~(mV)} &16&45&64&75&70&115\\
# {\rm error~(mV)}   &5&5&5&5&30&5\\
# \hline
# {\rm frequency~(Hz)} &70&80&90&100&110&\\
# {\rm voltage~(mV)} &142&167&183&160&221&\\
# {\rm error~(mV)}   &5&5&5&30&5&\\
# \hline
# \end{array} 
# \end{equation}
# 
# This data is also contained in the file 'linear_regression.csv'.
# 
# Required:
# <br>
# (i) Calculate the 4 elements of the curvature matrix.
# <br>
# (ii) Invert this to give the error matrix.
# <br>
# (iii) What are the uncertainties in the slope and intercept?
# <br>
# (iv) Comment on your answer.

# ### (i) Calculate the 4 elements of the curvature matrix.

# In[195]:


data = pd.read_csv('linear_regression.csv')
frequencies = data.iloc[:,0]
voltages = data.iloc[:,1]
voltage_errors = data.iloc[:,2]

def f(x, m, c):
    return m*x + c

def one_i():
    '''Your function should return something of the form np.matrix([[a_cc,a_cm],[a_mc,a_mm]])'''
    '''where m is the slope and c the intercept'''
    # YOUR CODE HERE
    popt, pcov = curve_fit(f, frequencies, voltages, sigma=voltage_errors)
    curvature_matrix = np.linalg.inv(pcov)

    return(curvature_matrix)

print(one_i())


# In[196]:


'''TEST CELL - DO NOT DELETE'''
assert isinstance(one_i(), (list, tuple, np.ndarray)), \
    'Please make sure that the return value is list/array of floats.'
assert len(one_i()) == 2 , 'Please make sure you return a matrix with dimensions 2x2' 


# ### (ii) Invert this to give the error matrix.

# In[197]:


data = pd.read_csv('linear_regression.csv')
frequencies = data.iloc[:,0]
voltages = data.iloc[:,1]
voltage_errors = data.iloc[:,2]

def one_ii():
    '''Your function should return something of the form np.matrix([[a_cc,a_cm],[a_mc,a_mm]])'''
    # YOUR CODE HERE
    popt, pcov = curve_fit(f, frequencies, voltages, sigma=voltage_errors)
    inverted_matrix=np.sqrt(np.diag(pcov))
    return(inverted_matrix)

print(one_ii())


# In[198]:


'''TEST CELL - DO NOT DELETE'''
assert isinstance(one_ii(), (list, tuple, np.ndarray)), \
    'Please make sure that the return value is list/array of floats.'
assert len(one_ii()) == 2 , 'Please make sure you return a matrix with dimensions 2x2' 


# ### (iii) What are the uncertainties in the slope and intercept?

# In[199]:


data = pd.read_csv('linear_regression.csv')
frequencies = data.iloc[:,0]
voltages = data.iloc[:,1]
voltage_errors = data.iloc[:,2]

def one_iii():
    slope_uncertainty = 0
    intercept_uncertainty = 0
    # YOUR CODE HERE
    popt, pcov = curve_fit(f, frequencies, voltages, sigma=voltage_errors)
    perr = np.sqrt(np.diag(pcov))
    slope_uncertainty = perr[0]
    intercept_uncertainty = perr[1]
    return(slope_uncertainty,intercept_uncertainty)

print(one_iii())


# In[200]:


'''TEST CELL - DO NOT DELETE'''
assert isinstance(one_iii(), (list, tuple, np.ndarray)), \
    'Please make sure that the return value is list/array of floats.'
assert len(one_iii()) == 2 , 'Please make sure you return a list/array of length 2' 


# In[201]:


'''TEST CELL - DO NOT DELETE'''


# ### (iv) Comment on your answer

# It can be clearly seen that the uncertainty of slope(0.05)  much more smaller than the uncertainty of intercept(3.41) for this data sets. The error of point 5 and point 10 much more bigger than others points, experiencer should test more datas to replace these two unaccurate points.

# ## Question 2: Using a calibration curve
# 
# A frequently encountered case where the correlation of the uncertainties must be taken into account is that of a calibration curve.  Consider the following set of measurements from an optical-activity experiment, where the angle of rotation of a plane-polarized light beam, $\theta$, is measured as a function of the independent variable, the concentration, $C$, of a sucrose solution. 
# 
# \begin{equation}
# \begin{array}{lcccc}
# \hline
# C \mbox{ (g cm$^{-3}$)} &0.025&0.05&0.075&0.100\\
# \theta \mbox{ (degrees)}&10.7&21.6&32.4&43.1\\
# \hline
# C \mbox{ (g cm$^{-3}$)}&0.125&0.150&0.175\\
# \theta \mbox{ (degrees)}&53.9&64.9&75.4\\
# \hline
# \end{array} 
# \end{equation}
# 
# The errors in the angle measurement are all $0.1^{\circ}$, the errors in the concentration are negligible.  A straight line  fit to the data yields  a gradient of $431.7\,^{\circ}\mbox{ g$^{-1}$ cm$^{3}$}$, and intercept $-0.03^{\circ}$. This data is contained in 'optical_activity.csv'.
# 
# <br>
# Required:
# <br>
# (i) Show that the curvature matrix, $\mathsf{A}$, is given by 
# 
# \begin{equation}
# \mathsf{A}=\left[\begin{array}{cc}
# 700\left((^{\circ})^{-2}\right)&70\left((^{\circ})^{-2}\mbox{g cm$^{-3}$}\right)\\
# 70\left((^{\circ})^{-2}\mbox{g cm$^{-3}$}\right)&8.75\left((\mbox{g/$^\circ$ cm$^{3})^2$}\right)\\
# \end{array}\right] ,
# \end{equation}
# 
# 
# and that the error matrix  is 
# 
# \begin{equation}
# \mathsf{C}=\left[\begin{array}{cc}
# 0.00714\left((^{\circ})^2\right)&-0.0571\left((^{\circ})^2\mbox{g$^{-1}$cm$^{3}$}\right)\\
# -0.0571\left((^{\circ})^2\mbox{g$^{-1}$cm$^{3}$}\right)&0.571\left((^{\circ})^2\mbox{g$^{-2}$ cm$^{6}$}\right)\\
# \end{array}\right] .
# \end{equation}
# 
# The entry for the intercept is in the top left-hand corner, that for the gradient in the bottom right-hand corner.  
# <br>
# (ii) Calculate the associated correlation matrix.  
# 
# Use the  entries of the error matrix to answer the following  questions: 
# <br>
# (iii) What are the uncertainties in the best-fit intercept and gradient? 
# <br>
# (iv) What optical rotation is expected for a known concentration of $C=0.080g cm^{-3}$, and what is the uncertainty? 
# <br>
# (v) What is the concentration given a measured rotation of $\theta=70.3^{\circ}$ and what is the uncertainty?

# ### (i) Verify the curvature matrix and the error matrix above.

# In[202]:


data = pd.read_csv('optical_activity.csv')
concentrations = data.iloc[:,0]
angles = data.iloc[:,1]
angle_errors = data.iloc[:,2]

def f(x, m, c):
    return m * x + c

def two_i():
    w = 1.0 / angle_errors**2

    A11 = np.sum(w)
    A12 = np.sum(w * concentrations)
    A22 = np.sum(w * concentrations**2)

    curvature_matrix = np.array([[A11, A12], [A12, A22]])
    error_matrix = np.linalg.inv(curvature_matrix)

    return curvature_matrix, error_matrix

print(two_i())


# In[203]:


'''TEST CELL - DO NOT DELETE'''
assert isinstance(two_i(), (list, tuple, np.ndarray)), \
    'Please make sure that the return value is list/array of floats.'
assert len(two_i()) == 2 , 'Please make sure you return a list/array of two matrices' 
assert len(two_i()[0]) == 2, 'Please make sure that the first entry is a 2x2 matrix' 
assert len(two_i()[1]) == 2, 'Please make sure that the second entry is a 2x2 matrix'  


# In[204]:


'''TEST CELL - DO NOT DELETE'''


# ### (ii) Calculate the associated correlation matrix.  

# In[205]:


data = pd.read_csv('optical_activity.csv')
concentrations = data.iloc[:,0]
angles = data.iloc[:,1]
angle_errors = data.iloc[:,2]

def two_ii():
    '''Your function should return something of the form np.matrix([[a_cc,a_cm],[a_mc,a_mm]])'''
    # YOUR CODE HERE
    curvature_matrix,error_matrix=two_i()
    C_11 = error_matrix[0, 0]
    C_22 = error_matrix[1, 1]
    C_12 = error_matrix[0, 1]
    C_21 = error_matrix[1, 0]

    R_11=1
    R_22=1
    R_12=C_12/np.sqrt(C_11*C_22)
    R_21=C_21/np.sqrt(C_11*C_22)
    return np.matrix([[R_11,R_12],[R_21,R_22]])
print(two_ii())    


# In[206]:


'''TEST CELL - DO NOT DELETE'''
assert isinstance(two_ii(), (list, tuple, np.ndarray)), \
    'Please make sure that the return value is a matrix.'
assert len(two_ii()) == 2 


# ### (iii) What are the uncertainties in the best-fit intercept and gradient? 

# In[207]:


data = pd.read_csv('optical_activity.csv')
concentrations = data.iloc[:,0]
angles = data.iloc[:,1]
angle_errors = data.iloc[:,2]

def two_iii():
    '''Your function should return the uncertainty in the gradient and intercept'''
    gradient_uncertainty = 0
    intercept_uncertainty = 0
    # YOUR CODE HERE
    _,error_matrix=two_i()
    intercept_uncertainty = np.sqrt(error_matrix[0, 0])
    gradient_uncertainty = np.sqrt(error_matrix[1, 1])
    return(gradient_uncertainty,intercept_uncertainty)

print(two_iii())


# In[208]:


'''TEST CELL - DO NOT DELETE'''
assert isinstance(two_iii(), (list, tuple, np.ndarray)), \
    'Please make sure that the return value is a list/array of floats.'
assert len(two_iii()) == 2 


# In[209]:


'''TEST CELL - DO NOT DELETE'''


# ### (iv) What optical rotation is expected for a known concentration of $C=0.080g cm^{-3}$, and what is the uncertainty? 

# In[210]:


data = pd.read_csv('optical_activity.csv')
concentrations = data.iloc[:,0]
angles = data.iloc[:,1]
angle_errors = data.iloc[:,2]

def get_best_fit(concentrations, angles, angle_errors):
    w = np.sum(1.0 / angle_errors**2)  # this is a scalar
    
    wx = np.sum(concentrations / angle_errors**2)
    wy = np.sum(angles / angle_errors**2)
    wxx = np.sum(concentrations**2 / angle_errors**2)
    wxy = np.sum(concentrations * angles / angle_errors**2)
    wyy = np.sum(angles**2 / angle_errors**2)
    
    denominator = w * wxx - wx**2
    
    m = (w * wxy - wx * wy) / denominator
    c = (wxx * wy - wx * wxy) / denominator
    
    return m, c

def two_iv():
    '''Your function should return the angle and the uncertainty'''
    angle = 0
    uncertainty = 0
    known_concentration=0.08
    # YOUR CODE HERE
    m, c = get_best_fit(concentrations, angles, angle_errors)
    rotation = m*known_concentration+c
    _,error_matrix = two_i()
    uncertainty = np.sqrt(error_matrix[0, 0] + 2 * error_matrix[0, 1] * known_concentration + error_matrix[1, 1] * known_concentration**2)
    
    return rotation,uncertainty
print(two_iv())


# In[211]:


'''TEST CELL - DO NOT DELETE'''
assert isinstance(two_iv(), (list, tuple, np.ndarray)), \
    'Please make sure that the return value is a list/array of floats.'
assert len(two_iv()) == 2 


# In[212]:


'''TEST CELL - DO NOT DELETE'''


# ### (v) What is the concentration given a measured rotation of $\theta=70.3^{\circ}$ and what is the uncertainty? You must return your answer in $gcm^{-3}$

# In[213]:


data = pd.read_csv('optical_activity.csv')
concentrations = data.iloc[:,0]
angles = data.iloc[:,1]
angle_errors = data.iloc[:,2]

def get_concentration_for_rotation(theta, m, c, error_matrix):
    # 计算浓度
    C = (theta - c) / m

    # 计算不确定性
    delta_theta = 1  # We consider a unit change in rotation
    delta_C = delta_theta / m

    # 计算浓度的不确定性
    uncertainty = np.sqrt(error_matrix[0, 0] + 2 * error_matrix[0, 1] * C + error_matrix[1, 1] * C**2) * delta_C

    return C, uncertainty

def two_v():
    '''Your function should return the concentration and uncertainty'''
    # YOUR CODE HERE
    m, c = get_best_fit(concentrations, angles, angle_errors)
    theta = 70.3
    _,error_matrix=two_i()
    concentration, uncertainty = get_concentration_for_rotation(theta, m, c, error_matrix)
    return(concentration,uncertainty)

print(two_v())


# In[214]:


'''TEST CELL - DO NOT DELETE'''
assert isinstance(two_v(), (list, tuple, np.ndarray)), \
    'Please make sure that the return value is a list/array of floats.'
assert len(two_v()) == 2 


# In[215]:


'''TEST CELL - DO NOT DELETE'''


# ## Question 3: Error bars from a $\chi^2$ minimisation to a non-linear function
# 
# In this question we will analyse the data shown in the figure below, which is an X-ray spectrum as a function of angle.
# 
# ![title](diffraction.JPG)
#  
# The data is contained in the file 'LorentzianData.csv'. There are three columns: the angle, the signal (in counts per second), and the error.  The number of X-rays counted in 20 seconds was recorded.
# 
# The model to describe the data has four parameters:  the height of the Lorentzian lineshape, $S_0$; the angle at which the peak is centered, $\theta_{0}$;
#  the angular width of the peak, $\Delta\theta$; and a constant background offset, $S_{\rm bgd}$. Mathematically, the signal, $S$, is of the form:
# \begin{equation}
# S=S_{\rm bgd}+\frac{S_{0}}{1+4\left(\frac{\theta-\theta_{0}}{\Delta\theta}\right)^2}.
# \end{equation}
# 
# and the function is defined by lorentzian(theta, s_0, s_bgd,delta_theta,theta_0).
# 
# Required:
# <br>
# (i) Explain how the error in the count rate was calculated.
# <br>
# (ii) Perform a $\chi^2$ minimisation.  What are the best-fit parameters?
# <br>
# (iii) Evaluate the error matrix.
# <br>
# (iv) Calculate the correlation matrix.
# <br>
# (v) What are the uncertainties in the best-fit parameters?
# <br>
# (vi) If you can plot contour plots, show the $\chi^2$ contours for 
# <br>
# >(a) background vs. peak centre. 
# <br>
# >(b) background vs. peak width.  
# 
# 
# These figures are shown in figure 6.11 of Hughes and Hase. Comment on the shape of the contours. Only your comment will be graded. 

# ### (i) Explain how the error in the count rate was calculated.

# The error in the count rate can be typically calculated based on the square root of the counts because it follows Poisson statistics.  If 
# N is the number of X-rays counted in a period, error(Standard deviation) is np.sqrt(N).

# ### (ii) Perform a $\chi^2$ minimisation.  What are the best-fit parameters?

# In[216]:


data = pd.read_csv('LorentzianData.csv') 

def lorentzian(theta, s_0, s_bgd,delta_theta,theta_0):
    return s_bgd+(s_0/(1+4*(((theta-theta_0)/delta_theta)**2)))

def residual(params):
    s_0, s_bgd, delta_theta, theta_0 = params
    angles = data.iloc[:, 0].values
    intensity = data.iloc[:, 1].values
    intensity_errors = data.iloc[:, 2].values
    residuals = (lorentzian(angles, s_0, s_bgd, delta_theta, theta_0) - intensity) / intensity_errors
    return np.sum(residuals**2)

def three_ii():
    s_0 = 0
    s_bgd = 0
    delta_theta = 0
    theta_0 = 0
    covariance_matrix = 0
    angles = data.iloc[:,0]
    intensity = data.iloc[:,1]
    intensity_errors = data.iloc[:,2]
    # YOUR CODE HERE
    initial_guess = [1, 1, 1, 1]
    result = minimize(residual, initial_guess)
    s_0, s_bgd, delta_theta, theta_0 = result.x
    
    _,covariance_matrix = curve_fit(lorentzian, angles, intensity, initial_guess, sigma=intensity_errors, method='trf')
    return (s_0,s_bgd,delta_theta,theta_0,np.matrix(covariance_matrix))

print(three_ii())


# In[217]:


'''TEST CELL - DO NOT DELETE'''
assert isinstance(three_ii(), (list, tuple, np.ndarray)), \
    'Please make sure that the return value is a list/array.'
assert len(three_ii()) == 5 , 'Please make sure that you return five values' 


# ### (iii) Evaluate the error matrix.

# In[218]:


data = pd.read_csv('LorentzianData.csv') 

def three_iii():
    '''Your function should return something of the form np.matrix([4x4])'''
    # YOUR CODE HERE
    s_0,s_bgd,delta_theta,theta_0,covariance_matrix = three_ii()
    error_matrix = np.sqrt(np.diag(covariance_matrix))
    return(error_matrix)

print(three_iii())


# In[219]:


'''TEST CELL - DO NOT DELETE'''
assert isinstance(three_iii(), (list, tuple, np.ndarray)), \
    'Please make sure that the return value is a matrix.'
assert len(three_iii()) == 4 , 'Please make sure that you return a 4x4 matrix' 


# ### (iv) Calculate the correlation matrix.

# In[220]:


data = pd.read_csv('LorentzianData.csv') 

def correlation_matrix(cov_matrix):
    std_devs = np.sqrt(np.diag(cov_matrix))
    correlation_matrix = np.empty_like(cov_matrix)
    
    for i in range(cov_matrix.shape[0]):
        for j in range(cov_matrix.shape[1]):
            correlation_matrix[i, j] = cov_matrix[i, j] / (std_devs[i] * std_devs[j])
            
    return correlation_matrix

def three_iv():
    '''Your function should return something of the form np.matrix([[a_cc,a_cm],[a_mc,a_mm]])'''
    # YOUR CODE HERE
    s_0,s_bgd,delta_theta,theta_0,covariance_matrix = three_ii()
    corr_matrix = correlation_matrix(covariance_matrix)
    return corr_matrix

print(three_iv())


# In[221]:


'''TEST CELL - DO NOT DELETE'''
assert isinstance(three_iv(), (list, tuple, np.ndarray)), \
    'Please make sure that the return value is a matrix.'
assert len(three_iv()) == 4 , 'Please make sure that you return a 4x4 matrix' 


# ### (v) What are the uncertainties in the best-fit parameters?

# In[224]:


data = pd.read_csv('LorentzianData.csv') 

def three_v():
    uncertainty_s_0 = 0
    uncertainty_s_bgd = 0
    uncertainty_delta_theta = 0
    uncertainty_theta_0 = 0
    # YOUR CODE HERE
    s_0,s_bgd,delta_theta,theta_0,covariance_matrix = three_ii()
    uncertainties = np.sqrt(np.diag(covariance_matrix))
    return uncertainties

print(three_v())


# In[225]:


'''TEST CELL - DO NOT DELETE'''


# ### (vi) These contours are shown in figure 6.11 of Hughes and Hase. Comment on the shape of the contours.

# As we can see the shape is oval, which means the correlation between parameters are closed. And it can be identify by the direction of the oval. Moreover, bigger size means bigger uncertainties. If some of the line overlap means data might repeat.

# ## Question 4: Prove the following properties:
# 
# Assume in this question that the uncertainties in $A$ and $B$ are correlated.
# <br>
# (i) If $Z=A\pm B$, show that
# ${\displaystyle\alpha_{Z}^2=\alpha_{A}^2+\alpha_{B}^2\pm2\alpha_{AB}}$.
# <br>
# (ii) If $Z=A\times B$, show that
#  ${\displaystyle\left(\frac{\alpha_Z}{Z}\right)^2=\left(\frac{\alpha_A}{A}\right)^2+\left(\frac{\alpha_B}{B}\right)^2+2\left(\frac{\alpha_{AB}}{AB}\right)}$.
# <br>
# (iii) If ${\displaystyle Z=\frac{A}{B}}$, show that
# ${\displaystyle\left(\frac{\alpha_Z}{Z}\right)^2=\left(\frac{\alpha_A}{A}\right)^2+\left(\frac{\alpha_B}{B}\right)^2-2\left(\frac{\alpha_{AB}}{AB}\right)}$.

# Variance formula: 𝛼2𝑍 = (∂Z/∂A)**2*𝛼2𝐴**2 + (∂Z/∂B)**2*𝛼2B**2 + 2*(∂Z/∂A)*(∂Z/∂B)*𝛼AB
# 
# (i) Z=A+B or A-B
# Because (∂Z/∂A)=1 and (∂Z/∂B)=1
# we have 𝛼2𝑍=𝛼2𝐴+𝛼2𝐵±2𝛼𝐴𝐵
# 
# (ii) Z = A*B
# Because (∂Z/∂A)=B and (∂Z/∂B)=A
# we have 𝛼2𝑍 = B**2*𝛼2𝐴**2 + A**2*𝛼2B**2 + 2*A*B*𝛼AB
# let we divide Z from both sides, because Z=A*B
# we have (𝛼𝑍/𝑍)2=(𝛼𝐴/𝐴)2+(𝛼𝐵/𝐵)2+2(𝛼𝐴𝐵/𝐴𝐵)
# 
# (iii) Z=A/b
# The same as i & ii, but (∂Z/∂A)=1/B, (∂Z/∂B)=-A/(B)**2
# we have (𝛼𝑍/𝑍)2=(𝛼𝐴/𝐴)2+(𝛼𝐵/𝐵)2−2(𝛼𝐴𝐵/𝐴𝐵)
