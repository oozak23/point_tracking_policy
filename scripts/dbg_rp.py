import sys, h5py, numpy as np, cv2
sys.path.insert(0,'/scratch/oozak23/droid_qp')
from droid_qp.robot_points import *
from droid_qp.geom import project, window_bounds
from droid_qp import video_io
vid='AUTOLab+7cb094d5+2023-12-14-00h-55m-09s__22008760'; ep,cam=vid.rsplit('__',1)
f=h5py.File(PW_FLOWS/f'{ep}_flows.h5','r')
w=sorted([k for k in f if ':' in k], key=lambda k:int(k.split(':')[0]))[0]
a,b=window_bounds(w); print('window',w)
E,K=window_calibration(f,w,cam)
pose=np.asarray(f[w]['gripper_pose'][:],dtype=np.float64)
jp=np.asarray(f[w]['joint_positions'][:],dtype=np.float64)
gp=np.asarray(f[w]['gripper_positions'][:],dtype=np.float64)
# 1) project gripper_pose (known correct 3D)
uvg,zg=project(pose[:,:3],E,K,out_size=(854,480))
print('gripper_pose uv[0..2]:', np.round(uvg[:3],1).tolist(),' z',np.round(zg[:3],2).tolist())
# 2) FK sampled points for the SAME window
s=load_sampler(128,seed=0)
p3=robot_points_3d(s,jp,gp); print('fk pts3 shape',p3.shape)
print('fk xyz range x[%.2f,%.2f] y[%.2f,%.2f] z[%.2f,%.2f]'%(p3[...,0].min(),p3[...,0].max(),p3[...,1].min(),p3[...,1].max(),p3[...,2].min(),p3[...,2].max()))
print('gripper_pose xyz     x[%.2f,%.2f] y[%.2f,%.2f] z[%.2f,%.2f]'%(pose[:,0].min(),pose[:,0].max(),pose[:,1].min(),pose[:,1].max(),pose[:,2].min(),pose[:,2].max()))
uv,ok=robot_points_2d(p3,E,K)
print('frame0: in-frame',int(ok[0].sum()),'/',p3.shape[1])
d=np.linalg.norm(uv[0]-uvg[0],axis=1)
print('dist of fk pts to projected flange (px): min %.0f p25 %.0f median %.0f'%(d.min(),np.percentile(d,25),np.median(d)))
