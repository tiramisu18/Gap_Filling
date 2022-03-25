import numpy as np
import scipy.io as scio


LandCover = np.load('./Simulation_Dataset/LandCover.npy')
Err = np.load('./Simulation_Dataset/Err_peren.npy')
LAI_Simu = np.load('./Simulation_Dataset/LAI_Simu_noErr.npy')
LAI_addErr = np.load('./Simulation_Dataset/LAI_Simu_addErr.npy')
Err_weight= np.load('./Simulation_Dataset/Err_weight.npy')

dataNew = 'LAI_Simulation.mat'
scio.savemat(dataNew, {'LAI_noErr': LAI_Simu, 'LAI_addErr': LAI_addErr, 'Err_percentage':Err})
dataNew = 'LandCover.mat'
scio.savemat(dataNew, {'LandCover': LandCover})