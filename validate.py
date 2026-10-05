from pathlib import Path
from typing import Sequence, Callable
import numpy as np
# Import JAX related packages
import jax
import jax.numpy as jnp
from jax import random, vmap
# Debugging packages; uncomment if needed
# from jax.config import config
# config.update("jax_enable_x64", True)
# config.update("jax_disable_jit", True)
# config.update("jax_debug_nans", True)
import pdb
# Flax imports
import flax
from flax import linen as nn
from flax.training import train_state
from flax import jax_utils

# Import Optax for optimization
import optax
# import pdb
# Import numpy and other utility libraries
import numpy as onp
import functools
import glob
import os, sys
import time

from train_old import load_model as load_model_old
from train import load_model, create_train_state

R_CUTOFF = 1.25
R_CUTOFF_SMOOTH = R_CUTOFF - 0.25
BATCH_SIZE = 16
BOUNDARY = 20

data_file = "./train_data.npz"
data = jnp.load(data_file)

params = {}
density = data["density"]
idx = np.where(data["density"].min(axis=1)>0)
density = density[idx[0]]
V_ext = data["V_ext"][idx[0]]
temp  = 2#data["temp"][idx[0]]
mu = 0.1 #data["mu"][idx[0]]
params["mu"] = mu#[...,None]
params["beta"] = 1/temp#[...,None]
c = jnp.log(density) + params["beta"]*V_ext - params["beta"]*params["mu"]
xyz = data["xyz"]


TRAIN_DATA = {
    "coord": xyz.reshape(80,80,80,3),
    "density": density[:-10,...].reshape(-1,80,80,80),
    "c": c[:-20,...].reshape(-1,80,80,80),
}

TEST_DATA = {
    "coord": xyz.reshape(80,80,80,3),
    "density": density[-10:,...].reshape(-1,80,80,80),
    "c": c[-20:-10,...].reshape(-1,80,80,80),
}

VALIDATION_DATA = {
    "coord": xyz.reshape(80,80,80,3),
    "density": density[-10:,...].reshape(-1,80,80,80),
    "c": c[-10:,...].reshape(-1,80,80,80),
}

def fcn_dist(x1,x2):
    dx = x1 - x2
    dx = dx - jnp.round(dx/BOUNDARY)*BOUNDARY
    return jnp.sqrt(jnp.sum(dx**2,axis=-1)),dx

dis_tmp,dis_vec = fcn_dist(TRAIN_DATA["coord"][0,0,0,:], TRAIN_DATA["coord"])
idx_tmp = jnp.where((dis_tmp < 2*R_CUTOFF+1e-6))
dis_idx = jnp.array(dis_tmp[idx_tmp])
dis_vec = jnp.array(dis_vec[idx_tmp[0],idx_tmp[1],idx_tmp[2],:])

def picard_density(indices, V_ext, B, mu, c1, a):

    snapshots = c1.shape[0]

    V_ext = V_ext[:snapshots, indices]
    # B = B
    # mu = mu

    rho = jnp.zeros(c1.shape)

    i = 0
    while i < 20000:
        rho_EL = jnp.exp((mu - V_ext)*B + c1)        
        rho = (1- a) * rho + a * rho_EL
        rho = rho.at[(~jnp.isfinite(rho)) | (rho < 0)].set(0)
        delta_p = jnp.max(np.abs(rho-rho_EL))

        threshold = delta_p/(jnp.mean(rho_EL))
        if i % 1000 == 0:
            print("% Difference:", threshold*100)
        
        if threshold < 1e-4: 
            return rho
        i += 1
    return None



def get_c1(state, ds, n, rng):

    ds_size = ds["coord"].shape[0]*ds["coord"].shape[1]*ds["coord"].shape[2] ### number of data points (volume)
    snapshot_size = ds["c"].shape[0]
    coord = ds["coord"]
    batch_size = BATCH_SIZE

    c1 = jnp.zeros((snapshot_size, n))
    indices = jax.random.permutation(rng, ds_size)[:n]

    voxels = indices.reshape((-1, batch_size))
    ## all_voxels = jnp.arange(ds_size).reshape((-1, batch_size)) ## all particles batched
    
    for snapshot_id in range(snapshot_size): ## system over time
        #print("snapshot_id:", snapshot_id, flush=True)s
        density = ds["density"][snapshot_id]

        c1_snap = jnp.zeros((n, ))

        for i in range(len(voxels)):
            batch = voxels[i]

            ## convert flattened array indices to voxel positions in 3d
            batch_x = batch // (coord.shape[1]*coord.shape[2])    
            batch_y = (batch % (coord.shape[1]*coord.shape[2])) // coord.shape[2]
            batch_z = batch % coord.shape[2]
            ## all neighboring particles 
            nei_idx  = (idx_tmp[0][None,:] + batch_x[..., None]) % coord.shape[0]
            nei_idy  = (idx_tmp[1][None,:] + batch_y[..., None]) % coord.shape[1]
            nei_idz  = (idx_tmp[2][None,:] + batch_z[..., None]) % coord.shape[2]

            rho_nei_batch = density[nei_idx,nei_idy,nei_idz]
            #print(batch_data["dis_vec"].shape, batch_data["rho_nei"].shape, flush=True)
            
            c1_vals = vmap(state.apply_fn,(None,None,0))(state.params, dis_vec, rho_nei_batch) #jax.vmap(apply_model_predict, axis_name="num_devices")(state, **batch_data)
            #print(c1_vals[:,0])
            c1_snap = c1_snap.at[i*batch_size:(i+1)*batch_size].set(c1_vals[:,0])

        c1 = c1.at[snapshot_id].set(c1_snap)

    return indices, c1
            

def main():
    def mse(c_pred, c_self):
        loss = jnp.mean((c_pred-c_self)**2, axis=-1)
        return loss, jnp.mean(loss), loss/jnp.mean(c_self**2, axis=-1), jnp.mean(loss/jnp.mean(c_self**2, axis=-1))
    
    def validate(model_path, load_fn):
        print(model_path)
        t, rng, state, train_error, test_error = load_fn(model_path)
        indices, c1 = get_c1(state, VALIDATION_DATA, 16000, jax.random.split(rng)[100])
        print("c1 shape:", c1.shape)
        snaps= c1.shape[0]
        # print(c[:snaps, indices])
        # print(c1)

        print("Mean Validation Loss:", mse(c[:snaps, indices], c1))
        # rho = picard_density(indices, V_ext, params["beta"], params["mu"], c1, 0.001)
        # #print(density[0, indices], rho[0])
        # print("Density MSE:", mse(rho, density[:snaps, indices]) if rho != None else None)
        # rho = picard_density(indices, 2*V_ext, params["beta"], params["mu"], c1, 0.03)
        # print(rho, density[:82, indices])
        print("#"*30)
        return
    validate(Path(f"model_save_path{R_CUTOFF}"), load_model_old)
    validate(Path(f"model_save_path{R_CUTOFF}_2"), load_model)
    validate(Path(f"model_save_path{R_CUTOFF}_5_2"), load_model)
    validate(Path(f"model_save_path{R_CUTOFF}_6_2"), load_model)
    validate(Path(f"model_save_path{R_CUTOFF}_7_4"), load_model)

main()
