"""
Optimized DeepMD code script.
"""

# Import common packages
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
from model import create_model
import time

# Checkpointing
#from orbax import checkpoints

# Network settings
FEATURE_NET_LAYER = np.array([12, 12, 24])
LEFT_FEATURE_LENGTH = FEATURE_NET_LAYER[-1]//2
ACTIVATION = jnp.tanh
DESCRIBE_NET_LAYER = np.array([64, 64, 64, 1])
R_CUTOFF = 1.25
R_CUTOFF_SMOOTH = R_CUTOFF - 0.25
TRAIN_ITERATION = 10000
BATCH_SIZE = 16
BOUNDARY = 20
SAVE_PATH = Path(f"model_save_path{R_CUTOFF}_7_4")

#print("hidden layer neurons:", FEATURE_NET_LAYER, DESCRIBE_NET_LAYER)
#print(LEFT_FEATURE_LENGTH)
print("model saved at: ", SAVE_PATH)
#SAVE_PATH.mkdir(parents=True, exist_ok=True)

def apply_model(state, dis_vec, rho_nei, c_self):
        def loss_fn(params):
            # jax.debug.print("Checking values: dis_vec={}", dis_vec)
            # jax.debug.print("Checking values: rho_nei={}", rho_nei)
            # jax.debug.print("Checking values: c_self={}", c_self)
            #print("shapes: ", dis_vec.shape, rho_nei.shape)
            force_predict = vmap(state.apply_fn,(None,None,0))(params, dis_vec[0], rho_nei)
            return jnp.sum((force_predict[:,0] - c_self)**2)
        
        loss_grad_fn = jax.value_and_grad(loss_fn)
        loss_val, grads = loss_grad_fn(state.params)
        grads = jax.lax.pmean(grads, axis_name="num_devices")
        return loss_val,grads

def apply_model_test(state, dis_vec, rho_nei, c_self):
    def loss_fn(params):

        force_predict = vmap(state.apply_fn,(None,None,0))(params, dis_vec[0], rho_nei)
        #print(force_predict[:,0])
        return jnp.sum((force_predict[:,0] - c_self)**2)
        #return jnp.sum((force_predict[:,0] - c_self)**2)/jnp.sum((c_self)**2)
    #print("cself:", c_self)
    loss_val = loss_fn(state.params)
    return loss_val,None

# Initialize the training state
@functools.partial(jax.pmap, static_broadcasted_argnums=(1),axis_name="esamble")
def create_train_state(rng, learning_rate):
    # learning_rate_fn = optax.exponential_decay(
    #     init_value=learning_rate,
    #     transition_steps=100000,
    #     decay_rate=0.9,
    #     end_value =1e-7
    # )
    print("Learning Rate:", learning_rate)
    # if epoch > 1000:
    #     learning_rate_fn = 1e-8
    # else:
    learning_rate_fn = optax.warmup_cosine_decay_schedule(
        init_value=1e-5,
        peak_value=learning_rate,
        warmup_steps=3000,
        decay_steps=57000,
        end_value=1e-8
    )
    # optax.join_schedules(
    #     schedules=[optax.linear_schedule(init_value=learning_rate, end_value=5e-5, transition_steps=1500),
    #     optax.cosine_decay_schedule(init_value=5e-5, decay_steps=200000, alpha=0.01)],
    #     boundaries=[1500]
    # )

    cf_init,cf_f,cf_e = create_model(
        FEATURE_NET_LAYER,
        DESCRIBE_NET_LAYER,
        fcn_dist,
        j_index,
        k_index,
        R_CUTOFF,
        R_CUTOFF_SMOOTH,
        LEFT_FEATURE_LENGTH
    )

    params = cf_init(rng)

    tx = optax.adamw(
        learning_rate=learning_rate_fn,
        b1=0.9,
        b2=0.9995,
        weight_decay=1e-4         
    )
    state = train_state.TrainState.create(
        apply_fn = cf_f,
        params = params,
        tx = tx
    )

    return state


params ={
}


data_file = "./train_data.npz"
data = jnp.load(data_file)


density = data["density"]
idx = np.where(data["density"].min(axis=1)>0)
density = density[idx[0]]
V_ext = data["V_ext"][idx[0]]
temp = 2 #data["temp"][idx[0]]
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
    "c": c[-10,...].reshape(-1,80,80,80),
}


def fcn_dist(x1,x2):
    dx = x1 - x2
    dx = dx - jnp.round(dx/BOUNDARY)*BOUNDARY
    return jnp.sqrt(jnp.sum(dx**2,axis=-1)),dx

dis_tmp,dis_vec = fcn_dist(TRAIN_DATA["coord"][0,0,0,:], TRAIN_DATA["coord"])
##idx_tmp are relative ids of voxels within
idx_tmp = jnp.where((dis_tmp < 2*R_CUTOFF+1e-6))
print("the first idx should be 0 0 0 :",idx_tmp[0][0],idx_tmp[1][0],idx_tmp[2][0])
dis_idx = jnp.array(dis_tmp[idx_tmp])
dis_vec = jnp.array(dis_vec[idx_tmp[0],idx_tmp[1],idx_tmp[2],:])
j_index = jnp.where((dis_idx < R_CUTOFF+1e-6))[0]
j_dis_vec = dis_vec[j_index]
dis_jk,dis_vec_jk = jax.vmap(fcn_dist,(0,None))(j_dis_vec, dis_vec)
k_index = []
for i in range(len(j_dis_vec)):
    k_index.append(jnp.where((dis_jk[i] < R_CUTOFF+1e-6))[0])
k_index = jnp.array(k_index)
print("j-Neighbors:", len(j_index))
print("k-Neighbors:", len(k_index))
# Update the model parameters

#print(type(dis_vec), type(density), type(c))

@jax.vmap
def update_model(state, grads):
    return state.apply_gradients(grads=grads)

# Training for an epoch
def train_epoch(state, train_ds, batch_size, t, rng):
    def train_helper(state, train_ds_snapshot, perms):
        perm_x, perm_y, perm_z = perms[0], perms[1], perms[2]
        nei_idx  = (idx_tmp[0][None,None,:] + perm_x[...,None]) % train_ds["coord"].shape[0]
        nei_idy  = (idx_tmp[1][None,None,:] + perm_y[...,None]) % train_ds["coord"].shape[1]
        nei_idz  = (idx_tmp[2][None,None,:] + perm_z[...,None]) % train_ds["coord"].shape[2]
        batch_data = {}
        batch_data["dis_vec"] = dis_vec[None,None,:,:]
        batch_data["rho_nei"] = train_ds_snapshot["density"][nei_idx,nei_idy,nei_idz]
        batch_data["c_self"] = train_ds_snapshot["c"][perm_x,perm_y,perm_z] 
        loss_val, grads = jax.vmap(apply_model, axis_name="num_devices")(state, **batch_data)
        state = update_model(state, grads)
        #epoch_loss.append(jax_utils.unreplicate(loss_val))
        return loss_val, state
    
        
    train_ds_size = train_ds["coord"].shape[0]*train_ds["coord"].shape[1]*train_ds["coord"].shape[2] ### number of data points (volume)
    snapshot_size = train_ds["c"].shape[0]
    steps_per_epoch = train_ds_size // batch_size
    # perms = jax.random.permutation(rng, train_ds_size)
    # perms = perms[:steps_per_epoch * batch_size].reshape((steps_per_epoch, batch_size)) ## the permutation is a random set of particle indices
    # print("train:", perms[:10])
    epoch_loss = []
    for snapshot_id in range(snapshot_size):
        train_ds_snapshot = {}
        train_ds_snapshot["density"] = train_ds["density"][snapshot_id]
        train_ds_snapshot["c"] = train_ds["c"][snapshot_id]
        train_ds_snapshot["coord"] = train_ds["coord"]
        if (state.step < 10000 and t % 5 == 0) or (state.step > 10000 and t % 10 == 0):
            for i in range(10):
                central_id = np.random.randint(0, train_ds_size)
                central_idx = central_id // (train_ds["coord"].shape[1]*train_ds["coord"].shape[2])    ## voxel indices to find the corresponding flat index
                central_idy = (central_id % (train_ds["coord"].shape[1]*train_ds["coord"].shape[2])) // train_ds["coord"].shape[2]
                central_idz = central_id % train_ds["coord"].shape[2]

                idx, idy, idz = jnp.meshgrid(jnp.array([central_idx-1, central_idx, central_idx+1]), jnp.array([central_idy-1, central_idy, central_idy+1]), jnp.array([central_idz-1, central_idz, central_idz+1]))
                idx = idx.reshape(jax.device_count(), -1) % train_ds["coord"].shape[0]
                idy = idy.reshape(jax.device_count(), -1) % train_ds["coord"].shape[1]
                idz = idz.reshape(jax.device_count(), -1) % train_ds["coord"].shape[2]

                loss_val, state = train_helper(state, train_ds_snapshot, [idx, idy, idz])
                epoch_loss.append(jax_utils.unreplicate(loss_val))
        # rng, input_rng = jax.random.split(rng)
        else:
            perms = np.random.randint(0, train_ds_size, 8*batch_size)
            perms = perms.reshape(-1, batch_size)
            for perm in perms:
                perm = perm.reshape(jax.device_count(), -1)
                perm_x = perm // (train_ds["coord"].shape[1]*train_ds["coord"].shape[2])    ## voxel indices to find the corresponding flat index
                perm_y = (perm % (train_ds["coord"].shape[1]*train_ds["coord"].shape[2])) // train_ds["coord"].shape[2]
                perm_z = perm % train_ds["coord"].shape[2]

                loss_val, state = train_helper(state, train_ds_snapshot, [perm_x, perm_y, perm_z])
                epoch_loss.append(jax_utils.unreplicate(loss_val))
                

    return state, onp.mean(epoch_loss)


# Training for an epoch
def test_epoch(state, train_ds, batch_size, rng):
    train_ds_size = train_ds["coord"].shape[0]*train_ds["coord"].shape[1]*train_ds["coord"].shape[2]
    snapshot_size = train_ds["c"].shape[0]
    # steps_per_epoch = train_ds_size // batch_size
    # perms = jax.random.permutation(rng, train_ds_size)
    # perms = perms[:steps_per_epoch * batch_size].reshape((steps_per_epoch, batch_size))
    epoch_loss = []
    #print("test:", perms[:10])
    for snapshot_id in range(snapshot_size):
        train_ds_snapshot = {}
        train_ds_snapshot["density"] = train_ds["density"][snapshot_id]
        train_ds_snapshot["c"] = train_ds["c"][snapshot_id]
        train_ds_snapshot["coord"] = train_ds["coord"]
        perms = np.random.randint(0, train_ds_size, 10*batch_size)
        perms = perms.reshape(-1, batch_size)
        for perm in perms:
            perm = perm.reshape(jax.device_count(), -1)
            perm_x = perm // (train_ds["coord"].shape[1]*train_ds["coord"].shape[2])    
            perm_y = (perm % (train_ds["coord"].shape[1]*train_ds["coord"].shape[2])) // train_ds["coord"].shape[2]
            perm_z = perm % train_ds["coord"].shape[2]
            nei_idx  = (idx_tmp[0][None,None,:] + perm_x[...,None]) % train_ds["coord"].shape[0]
            nei_idy  = (idx_tmp[1][None,None,:] + perm_y[...,None]) % train_ds["coord"].shape[1]
            nei_idz  = (idx_tmp[2][None,None,:] + perm_z[...,None]) % train_ds["coord"].shape[2]
            batch_data = {}
            batch_data["dis_vec"] = dis_vec[None,None,:,:]
            batch_data["rho_nei"] = train_ds_snapshot["density"][nei_idx,nei_idy,nei_idz]
            batch_data["c_self"] = train_ds_snapshot["c"][perm_x,perm_y,perm_z] 
            loss_val, grads = jax.vmap(apply_model_test, axis_name="num_devices")(state, **batch_data)
            epoch_loss.append(jax_utils.unreplicate(loss_val))

    return onp.mean(epoch_loss)


# Initialize the random number generator
# rng = jax.random.PRNGKey(0)
# rng, init_rng = jax.random.split(rng)

# state = create_train_state(init_rng[None].repeat(jax.device_count(),axis=0), 1e-5)
# train_error = onp.zeros(TRAIN_ITERATION)
# #state = jax_utils.replicate(state)

def load_model(path):
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    state = create_train_state(init_rng[None].repeat(jax.device_count(),axis=0), 6e-4)
    if os.path.exists(path) and len(os.listdir(path)) > 0:
        list_dir = np.array(os.listdir(path))
        mask = np.array(list(map(lambda x: x.startswith("jax_ckpt"), list_dir)))
        file = np.sort(list_dir[mask])[-1]
        model_save = jnp.load(path/file, allow_pickle=True)
        t, params = model_save["t"], model_save["params"].item()
        print("Epochs Trained", t)
        state = state.replace(params=jax.tree_util.tree_map(jnp.asarray, params))
    
        train_error = onp.load(path/ "train_error.npy")
        test_error = onp.load(path/ "test_error.npy")
        #errors = #errors[errors > 0]
    else:
        SAVE_PATH.mkdir(parents=True, exist_ok=True)
        train_error, test_error = onp.zeros(TRAIN_ITERATION), onp.zeros(TRAIN_ITERATION)
        t=-1

    return t, rng, state, train_error, test_error


#Start training
def train_loop(rng, state, train_error, test_error, epoch_save=-1, epoch_num=TRAIN_ITERATION):

    start = time.time()
    for t in range(epoch_save+1, epoch_num+epoch_save):
        rng, input_rng = jax.random.split(rng)
        state, loss_val = train_epoch(state, TRAIN_DATA,BATCH_SIZE, t, input_rng)
        train_error[t] = loss_val
        if t % 1 == 0:
            error_val = test_epoch(state, TEST_DATA,BATCH_SIZE,input_rng)
            test_error[t]=error_val
            print('step: {}, loss: {}, error {}'.format(state.step,loss_val,error_val), flush=True)

        if epoch_num > 10 and t % (epoch_num//10) == 0:

            onp.save(SAVE_PATH/ "train_error.npy",train_error)
            onp.save(SAVE_PATH/ "test_error.npy", test_error)
            
            list_dir = np.array(os.listdir(SAVE_PATH))
            mask = np.array(list(map(lambda x: x.startswith("jax_ckpt"), list_dir)))
            file_path = np.sort(list_dir[mask])
            if len(file_path) >= 3:
                os.remove(SAVE_PATH/ file_path[0])

            ckpt_filename = SAVE_PATH / f'jax_ckpt_{t:06d}.npz'
            params = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state.params))
            #opt_state=onp.array(state.opt_state, dtype=object)
            with open(ckpt_filename, 'wb') as f:
                jnp.savez(f,t=t,params=params)
            #error_val = test_epoch(state, TEST_DATA,BATCH_SIZE,input_rng)   

        if epoch_num > 10 and t % (epoch_num//10) == 0:
            end = time.time()
            print(f"Training {epoch_num//10 if epoch_num > 10 else 1} Epoch:  {np.round(end-start, 3)}", flush=True)
            start = time.time()
                

if __name__ == "__main__":

    t, rng, state, train_error, test_error = load_model(SAVE_PATH)
    if onp.count_nonzero(train_error) != 0:
        state = state.replace(params=jax.tree_util.tree_map(lambda x: x[None], state.params))
    train_loop(rng, state, train_error, test_error, epoch_save=t,epoch_num=1001)
    #print(state.params)
