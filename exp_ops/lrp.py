import sys, re, copy
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from tensorflow.python.ops import nn_ops, gen_nn_ops
import matplotlib.pyplot as plt

def lrp(F, lowest, highest, graph=None, return_flist=False):
    #Assumptions:
        #all conv strides are [1,1,1,1]
        #all pool strides are [1,2,2,1]
        #all pool/conv padding is SAME
        #only reshaping that happens is after a pool layer (pool -> fc) or a conv layer (conv -> fc)
    F_list = []
    traversed, graph, graph_dict, var_dict = get_traversed(graph=graph)
    for n in traversed:
        val_name = next(I for I in graph_dict[n].input if I in traversed).split("/read")[0] + ":0"
        X = graph.get_tensor_by_name(val_name)      
        if graph_dict[n].op == "MatMul":
            weight_name = next(I for I in graph_dict[n].input if not I in traversed).split("/read")[0] + ":0"
            W = var_dict[weight_name]
            if "absolute_input" in graph_dict[n].input:
                F = fprop_first(F, W, X, lowest, highest)
                F_list.append(F)
                break
            else:
                F = fprop(F, W, X) 
                F_list.append(F)
        elif graph_dict[n].op == "MaxPool" or graph_dict[n].op ==  "MaxPoolWithArgmax":
            F = fprop_pool(F, X)     
            F_list.append(F)
        elif graph_dict[n].op == "Conv2D":
            weight_name = next(I for I in graph_dict[n].input if not I in traversed).split("/read")[0] + ":0"
            W = var_dict[weight_name]
            if "absolute_input" in graph_dict[n].input:
                F = fprop_conv_first(F, W, X, lowest, highest)
                F_list.append(F)
                break
            else:
                F = fprop_conv(F, W, X) 
                F_list.append(F)
    if return_flist:
        return F_list
    else:
        return F

def get_traversed(graph = None):
    graph = tf.get_default_graph() if graph is None else graph
    graph_dict = {node.name:node for node in graph.as_graph_def().node}
    var_dict = {v.name:v.value() for v in tf.get_collection(tf.GraphKeys.VARIABLES)}
    return traverse(graph_dict["absolute_output"], [], graph_dict), graph, graph_dict, var_dict

def traverse(node, L, graph_dict):
    #Depth First Search the Network Graph
    L.append(node.name)
    if "absolute_input" in node.name:
        return L
    inputs = node.input

    for nodename in inputs:
        if not traverse(graph_dict[nodename], L, graph_dict) is None:
            return L
    return None

def fprop_first(F, W, X, lowest, highest):
    W,V,U = W,tf.maximum(0.0,W), tf.minimum(0.0,W)
    X,L,H = X, X*0+lowest, X*0+highest

    Z = tf.matmul(X, W)-tf.matmul(L, V)-tf.matmul(H, U)+1e-9
    S = F/Z
    F = X*tf.matmul(S,tf.transpose(W))-L*tf.matmul(S, tf.transpose(V))-H*tf.matmul(S,tf.transpose(U))
    return F

def fprop(F, W, X):
    V = tf.maximum(0.0, W)
    Z = tf.matmul(X, V)+1e-9;
    S = F/Z
    C = tf.matmul(S, tf.transpose(V))        
    F = X*C
    return F

def fprop_conv_first(F, W, X, lowest, highest, strides=None, padding='SAME'):
    strides = [1, 1, 1, 1] if strides is None else strides

    Wn = tf.minimum(0.0, W)
    Wp = tf.maximum(0.0, W)

    X, L, H = X, X*0+lowest, X*0+highest

    c  = tf.nn.conv2d(X, W, strides, padding)
    cp = tf.nn.conv2d(H, Wp, strides, padding)
    cn = tf.nn.conv2d(L, Wn, strides, padding)
    Z = c - cp - cn + 1e-9
    S = F/Z
    
    g  = nn_ops.conv2d_backprop_input(tf.shape(X), W,  S, strides, padding)
    gp = nn_ops.conv2d_backprop_input(tf.shape(X), Wp, S, strides, padding)
    gn = nn_ops.conv2d_backprop_input(tf.shape(X), Wn, S, strides, padding)
    F = X*g - L*gp - H*gn
    return F

def fprop_conv(F, W, X, strides=None, padding='SAME'):
    xshape = X.get_shape().as_list()
    fshape = F.get_shape().as_list()
    if len(xshape) != len(fshape):
        F = tf.reshape(F, (-1, xshape[1], xshape[2], fshape[-1]/(xshape[1]*xshape[2])))
    strides = [1, 1, 1, 1] if strides is None else strides
    W = tf.maximum(0.0, W)

    Z = tf.nn.conv2d(X, W, strides, padding) + 1e-9 
    S = F/Z
    C = nn_ops.conv2d_backprop_input(tf.shape(X), W,  S, strides, padding)
    F = X*C
    return F

def fprop_pool(F, X, strides=None, ksize=None, padding='SAME'):
    xshape = X.get_shape().as_list()
    fshape = F.get_shape().as_list()
    if len(xshape) != len(fshape):
        F = tf.reshape(F, (-1, int(np.ceil(xshape[1]/2.0)), 
                               int(np.ceil(xshape[2]/2.0)), xshape[3]))
    ksize = [1, 2, 2, 1]  if ksize is None else ksize
    strides = [1, 2, 2, 1]  if strides is None else strides

    Z = tf.nn.max_pool(X, strides=strides, ksize=ksize, padding=padding) + 1e-9
    S = F / Z
    C = gen_nn_ops._max_pool_grad(X, Z, S, ksize, strides, padding)    
    F = X*C
    return F


def get_lrp_im(sess, F, x, y, xval, yval):
    im = []
    for i in range(0, xval.shape[0]):
        im += list(F.eval(session=sess, feed_dict={x: xval, y: yval}))
    return im

def visualize(sess, F, x, y, xval, yval, bsize=2):
    im = []
    for i in range(0, xval.shape[0], bsize):
        im += list(F.eval(session=sess, feed_dict={x: xval, y: yval}))

    rgb = []        
    for i in range(len(im)):
        r = np.dstack((im[i][:,:,2],im[i][:,:,2],im[i][:,:,2]))
        g = np.dstack((im[i][:,:,1],im[i][:,:,1],im[i][:,:,1]))
        b = np.dstack((im[i][:,:,0],im[i][:,:,0],im[i][:,:,0]))
        rgb = np.append(rgb,np.concatenate((r[:,:,None],g[:,:,None],b[:,:,None]),axis=2))
    return rgb
