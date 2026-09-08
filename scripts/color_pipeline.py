"""Explicit linear Rec.709 -> absolute HDR Rec.2020 -> ST2084 transforms."""
import numpy as np
REC709_TO_2020=np.array([[.6274038959,.3292830384,.0433130657],[.0690972894,.9195403951,.0113623156],[.0163914389,.0880133079,.8955952532]],dtype=np.float32)
LUMA709=np.array([.2126,.7152,.0722],np.float32)
LUMA2020=np.array([.2627,.6780,.0593],np.float32)

def srgb_to_linear(x):
    return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)

def linear_to_srgb(x):
    x=np.maximum(x,0)
    return np.where(x<=.0031308,12.92*x,1.055*x**(1/2.4)-.055)

def grade_nits(rgb, exposure=1.):
    # 1.0 scene-linear -> 200 nits before highlight rolloff.
    # Real CG emissions retain values >1, generating real HDR highlights.
    nits=np.maximum(rgb,0)*(200*exposure)
    peak=nits.max(axis=-1,keepdims=True)
    rolled=np.where(peak<=200,peak,200+800*(-np.expm1(-np.maximum(peak-200,0)/800)))
    return nits*(rolled/np.maximum(peak,1e-8))

def pq_encode(nits):
    m1=2610/16384;m2=2523/32;c1=3424/4096;c2=2413/128;c3=2392/128
    p=(np.maximum(nits,0)/10000)**m1
    return ((c1+c2*p)/(1+c3*p))**m2

def pq_decode(code):
    m1=2610/16384;m2=2523/32;c1=3424/4096;c2=2413/128;c3=2392/128
    p=np.maximum(code,0)**(1/m2)
    return 10000*(np.maximum(p-c1,0)/np.maximum(c2-c3*p,1e-8))**(1/m1)

def sdr_preview(rgb):
    # Explicit inspection-only SDR tone map of the absolute HDR grade.
    n=grade_nits(rgb)
    lum=n@LUMA709
    mapped=lum/(lum+95)
    out=n*(mapped/np.maximum(lum,1e-6))[...,None]
    return np.clip(linear_to_srgb(out),0,1)
