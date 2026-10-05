"""
Code to implement basic DeepMD functionalities.
"""

# Common packages
import numpy as np
import glob
#from absl import logging
import datetime
import os
import zipfile
from typing import Sequence, Optional
from functools import partial
import pdb
# Jax packages
import jax
import jax.numpy as jnp
from jax import vmap
from jax.lax import stop_gradient

# Flax packages
import flax.linen as nn
import flax
from flax.training import train_state  # Useful dataclass to keep train state
from flax import struct  # Flax dataclasses


class MLP(nn.Module):
    """
    Simple MLP (Multilayer Perceptron) function copied from Flax
    """
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = jax.nn.silu(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1])(x)
        return x

def create_model(feature_net_layer, describe_net_layer,fcn_dist,j_index,k_index,
                 r_cutoff, r_cutoff_smooth, left_feature_length):
    """
    Function to create the model
    """

    #feature_net_mlp = MLP(feature_net_layer)
    feature_net_left = MLP(feature_net_layer//2)
    print("Left Feat Network: ", feature_net_layer//2)
    feature_net_right = MLP(feature_net_layer)
    print("Right Feat Network: ", feature_net_layer)
    describe_net_mlp = MLP(describe_net_layer)
    print("Describe Network: ", describe_net_layer)

    def init(key):
        params = {}
        key1, key_tmp = jax.random.split(key)
        key2, key3 = jax.random.split(key_tmp)
        batch = jnp.ones((32, 3))
        params["feature_net_left_param"] = feature_net_left.init(key1, batch)
        params["feature_net_right_param"] = feature_net_right.init(key2, batch)
        #batch = jnp.ones((16, left_feature_length*(feature_net_layer[-1]-left_feature_length)))
        batch = jnp.ones((32, feature_net_layer[-1]*feature_net_layer[-1]//2))
        params["describe_net_param"] = describe_net_mlp.init(key3, batch)
        return params

    displacement_vmap = None

    @jax.jit
    def fcn_feature(coord, idx):
        dr = displacement_vmap(coord, coord[idx, :])
        dr = jnp.where((idx >= coord.shape[0])[..., None], (r_cutoff + 3), dr)
        return dr

    @jax.jit
    def fcn_e(potential_params, pos,rho_nei):
        pos_j = pos[j_index]
        pos_k = pos[k_index]
        dis,dis_vec = jax.vmap(fcn_dist)(pos_j,pos_k)
        rho_nei_j = rho_nei[j_index]
        rho_nei_k = rho_nei[k_index]
        energy = fcn_ei(potential_params, dis_vec, rho_nei_j, rho_nei_k)
        return jnp.sum(energy)*(20/80)**3

    @jax.jit
    def fcn_ei(potential_params, dr, rho_nei_j, rho_nei_k):
        """
        Returns the energy of each atom.

        Args:
            potential_params: network parameters
            dr: distance vector of the atoms in cutoff

        Returns:
            Ei: energy of each element
        """
        print(dr.shape, rho_nei_j.shape, rho_nei_k.shape)
        d = jnp.sqrt(jnp.sum(dr**2,axis=-1)+1e-6)[..., None] 
        smooth = jnp.where(d < r_cutoff_smooth, 1, 0.5 * (jnp.cos(jnp.pi * (d - r_cutoff_smooth) / (r_cutoff - r_cutoff_smooth)) + 0.5))
        smooth = jnp.where(d > r_cutoff, 0, smooth)
        sd = smooth / d #(N, N, 1)
        sdr = sd * dr / d #(N, N, 3)
        
        R_i = jnp.concatenate([sd,sdr],axis=-1) #(N, N, 4)
        input = jnp.concatenate([rho_nei_j[...,None,None].repeat(sd.shape[1],axis=-2),rho_nei_k[...,None],sd],axis=-1) ##concatenating neighbors and smooth into one tensor to pass into MLP
        left_feature = feature_net_left.apply(potential_params["feature_net_left_param"], input) #(N, N, features[-1])
        right_feature = feature_net_right.apply(potential_params["feature_net_right_param"], input) #(N, N, features[-1])
        #feature = feature_net_mlp.apply(potential_params["feature_net_param"], input)
        #print(left_feature.shape, right_feature.shape)
        
        left_feature = jnp.sum(left_feature[..., None] * R_i[..., None, :], axis=1)*(20/80)**3 #(N, features[-1], 4)
        right_feature = jnp.sum(right_feature[..., None] * R_i[..., None, :], axis=1)*(20/80)**3 #(N, features[-1], 4)
        #print(left_feature.shape, right_feature.shape)
        
        # left_feature = jnp.sum(feature[..., :left_feature_length, None] * R_i[..., None, :], axis=-3)*(20/80)**3 #(N, left_feature_length, 4)
        # right_feature = jnp.sum(feature[..., left_feature_length:, None]* R_i[..., None, :], axis=-3)*(20/80)**3 #(N, features[-1]-left_feature_length, 4)
        
        describe = jnp.sum(left_feature[...,None,:] * right_feature[...,None,:,:], axis=-1).reshape([left_feature.shape[0],-1])
        #describe = jnp.sum(left_feature * right_feature, axis=-1).reshape([left_feature.shape[0],-1])

        energy = describe_net_mlp.apply(potential_params["describe_net_param"], describe)
        return energy

    grad_e = jax.grad(fcn_e, argnums=2)
    grad_ei = jax.grad(fcn_ei, argnums=1) 

    @jax.jit
    def fcn_f(*x):
        return -grad_e(*x) ## F = -∇U

    @jax.jit
    def fcn_fij(*x):
        logging.error("Incorrect definition. Maybe useful in the future.")
        return -grad_ei(*x)

    return init, fcn_f, fcn_e
