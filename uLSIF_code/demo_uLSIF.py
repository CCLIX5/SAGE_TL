import numpy as np
import uLSIF as ul
import pdf_Gaussian as pdfG
import matplotlib.pyplot as plt
import pandas as pd
np.random.seed(0)

d = 1
dataset = 1
if dataset == 1:
    n_de = 100
    n_nu = 100
    mu_de = 1
    mu_nu = 1
    sigma_de = 1 / 2
    sigma_nu = 1 / 8
    legend_position = 'upper right'
elif dataset == 2:
    n_de = 200
    n_nu = 1000
    mu_de = 1
    mu_nu = 2
    sigma_de = 1 / 2
    sigma_nu = 1 / 4
    legend_position = 'upper left'

# Generate data for domain adaptation
x_de = np.random.normal(loc=mu_de, scale=sigma_de, size=(d, n_de))
x_nu = np.random.normal(loc=mu_nu, scale=sigma_nu, size=(d, n_nu))

x_disp = np.array([np.linspace(-0.5,3,100)])

p_de_x_disp = pdfG.pdf_Gaussian(x_disp,mu_de,sigma_de)
p_nu_x_disp = pdfG.pdf_Gaussian(x_disp,mu_nu,sigma_nu)
w_x_disp=p_nu_x_disp/p_de_x_disp

p_de_x_de = pdfG.pdf_Gaussian(x_de,mu_de,sigma_de)
p_nu_x_de = pdfG.pdf_Gaussian(x_de,mu_nu,sigma_nu)
w_x_de=p_nu_x_de/p_de_x_de

wh_x_de,wh_x_disp =ul.uLSIF(x_de,x_nu,x_disp,np.logspace(-3, 1, 9),np.logspace(-3, 1, 9),100,5)


plt.figure(1)
plt.plot(x_disp[0], p_de_x_disp, "b")    
plt.plot(x_disp[0], p_nu_x_disp, "r")    
plt.legend(loc=legend_position,labels=['p_de(x)','p_nu(x)'])

plt.figure(2)
plt.plot(x_disp[0], w_x_disp, "b")    
plt.plot(x_disp[0], wh_x_disp, "r")  
plt.scatter(x_de[0], wh_x_de)
plt.legend(loc=legend_position,labels=['w(x)','w_hat(x)','w-hat(x^de)'])

plt.show() 

