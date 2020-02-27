"""
Created on Wed Feb 12 17:04:56 2020
 
@author: paulo
"""
 
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import pandas as pd
import scipy as scipy
from mpl_toolkits.mplot3d import Axes3D
 
dataset = pd.read_csv('laser_penetration_depth_study.csv', sep = ';')
 
 
dataset0 = dataset.astype(float)
 
dataset_t = dataset0[~np.isnan(dataset0)].dropna(axis='rows')
table = dataset_t.pivot('current (mA)','time (s)','AFM height (nm)')
 
 
import seaborn as sns
sns.set(font_scale=1.4)
ax = sns.heatmap(table, annot=True, fmt=".2f",annot_kws={"size": 18},cbar_kws={'label': 'Gold thicknes (nm)'})
ax.figure.axes[-1].yaxis.label.set_size(18)
# fix for mpl bug that cuts off top/bottom of seaborn viz
b, t = plt.ylim() # discover the values for bottom and top
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # Subtract 0.5 from the top
plt.title('Gold thickness variation with different time and current settings @ 0.08 mbar',fontsize=18)
plt.ylim(b, t) # update the ylim(bottom, top) values
plt.show() # ta-da!
 
 
# some 3-dim points
t = np.array(list(table.columns))
I = np.array(list(table.index))
height = table.values.flatten()
 
 
# regular grid covering the domain of the data
X,Y = np.meshgrid(t, I)
XX = X.flatten()
YY = Y.flatten()
 
data = np.transpose([XX,YY,height])
 
A = np.c_[np.ones(data.shape[0]), data[:,:2], np.prod(data[:,:2], axis=1), data[:,:2]**2]
C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])
 
# evaluate it on a grid
Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2], C).reshape(X.shape)
 
height_fit = Z.flatten()
 
ss_res = np.sum((height - height_fit) ** 2)       # residual sum of squares
ss_tot = np.sum((height - np.mean(height)) ** 2)  # total sum of squares
r2 = 1 - (ss_res / ss_tot)              # R squared fit, R^2
 
# plot points and fitted surface
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_title('Gold thickness variation with time and current')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
ax.scatter(data[:,0], data[:,1], data[:,2], c='r', s=50,label='Au thickness (t,I)')
ax.text2D(0.05, 0.8,'$R^{2}=%.3f$'%(round(r2,3)), fontsize=15, transform=ax.transAxes)
ax.text2D(0.05, 0.75, '$h(t,I)=%.3f+(%.3f)t+(%.3f)I+(%.3f)t*I+(%.3f)t^{2}+(%.3f)I^{2}$'%(C[0],C[1],C[2],C[3],C[4],C[5]), fontsize=15, transform=ax.transAxes)
ax.set_xlabel('time (s)')
ax.set_ylabel('Current (mA)')
ax.set_zlabel('Au thickness (nm)')
plt.legend(loc='best')
#ax.axis('equal')
#ax.axis('tight')
plt.show()
 
 
 
#################################################################
#PENETRATION DEPTH
#################################################################
 
 
height_pred = C[0]+C[1]*100+C[2]*40+C[3]*100*40+C[4]*100**2+C[5]*40**2
height_pred_e = 1
 
raman_s = np.array(dataset0['RAMAN intensity SI'])
raman_s_e = np.array(dataset0['e'])
height_e = np.array(dataset0['e (nm)'].fillna(value=height_pred_e))
height = np.array(dataset0['AFM height (nm)'].fillna(value=height_pred))
 
y = np.log(raman_s[:-2])
x= height[:-2]
#e_x = np.log(height_e)
 
M = x[:,np.newaxis]**[0,1]
 
p,_,_,_ = scipy.linalg.lstsq(M,y)
 
height_array = np.linspace(x.min(),x.max(),100)
 
 
pen = p[0] + p[1]*x
pen_array = p[0]+p[1]*height_array
 
#popt, pcov = scipy.optimize.curve_fit(lambda t,a,b: a*np.exp(b*t),  height,  raman_s,  p0=(4, 0.1))
 
 
ss_res = np.sum((y - pen) ** 2)       # residual sum of squares
ss_tot = np.sum((y - np.mean(y)) ** 2)  # total sum of squares
r2 = 1 - (ss_res / ss_tot)              # R squared fit, R^2
 
fig = plt.figure()
ax = plt.subplot()
ax.set_title('Laser penetration depth on a Gold coated silicon die')
#ax.errorbar(height, raman_s,yerr=raman_s_e, xerr=height_e ,c='r', fmt='o',elinewidth=3, capsize=0)
ax.errorbar(height,np.log(raman_s),xerr=height_e,yerr=np.log(raman_s_e)/raman_s, c='r',fmt='o',label='ln(Intensity(Au thickness))')
ax.plot(height_array, pen_array,'b--')
ax.text(0.7, 0.9,'$R^{2}=%.3f$'%(round(r2,3)), fontsize=15, transform=ax.transAxes,c='b')
ax.text(0.7, 0.85, '$ln(I(h))=ln(I(0))-a*h=%.1f %.3f*h$'%(p[0],p[1]), fontsize=15, transform=ax.transAxes,c='b')
ax.set_xlabel('Gold thickness (nm)')
ax.set_ylabel('ln(Intensity (a.u.))')
ax.axis('equal')
ax.axis('tight')
plt.legend(loc='best')
plt.show()
 
 
 
 
###############################################################################
#AREA
###############################################################################
 
 
from sklearn.linear_model import LinearRegression
 
area = np.array(dataset0['baseline area'])
area_m = area/area.min()
 
raman_s_m = raman_s/raman_s.min()
 
 
fig = plt.figure()
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)
 
linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(height.reshape(-1,1), area_m.reshape(-1,1))  # perform linear regression
Y_pred = linear_regressor.predict(height.reshape(-1,1))  # make predictions
 
ax1.scatter(height, area_m, c='r', s=50,label='Baseline area(height)/min(baseline)')
ax1.plot(height, Y_pred, 'r--',label='regression line')
ax1.set_xlabel('Gold thickness (nm)')
ax1.set_ylabel('Area under the curve (a.u.*cm-1)')
ax1.axis('equal')
ax1.axis('tight')
ax1.legend(loc='best')
 
 
linear_regressor2 = LinearRegression()  # create object for the class
linear_regressor2.fit(raman_s_m.reshape(-1,1), area_m.reshape(-1,1))  # perform linear regression
Y_pred2 = linear_regressor2.predict(raman_s_m.reshape(-1,1))  # make predictions
 
 
ax2.scatter(raman_s_m, area_m, c='b', s=50,label='Baseline area(Si)/min(baseline)')
ax2.plot(raman_s_m, Y_pred2, 'b--',label='regression line')
ax2.set_xlabel('Silicon peak/min(silicon peak)')
ax2.set_ylabel('Area under the curve (a.u.*cm-1)')
ax2.axis('equal')
ax2.axis('tight')
 
ax2.legend(loc='best')
plt.show()
