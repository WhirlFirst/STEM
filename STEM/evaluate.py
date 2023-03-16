import pickle
import numpy as np
import pandas as pd
import scipy

def k_coord(mappingmtx,spcoor, k=1):
    maskmap = np.zeros_like(mappingmtx)

    argmask = np.argsort(1/mappingmtx.values,axis=1)

    for i in range(argmask.shape[0]):
        for j in range(k):
            maskmap[i,argmask[i,j]]=1

    mappingnorm = (maskmap*mappingmtx).div((maskmap*mappingmtx).sum(axis=1), axis=0)

    spaotsc_coord = np.zeros([mappingnorm.shape[0],2])
    spaotsc_coord[:,0] = np.matmul(mappingnorm.values,spcoor.coord_x.values)
    spaotsc_coord[:,1] = np.matmul(mappingnorm.values,spcoor.coord_y.values)
    return spaotsc_coord

def all_coord(mappingmtx,spcoor):
    mappingnorm = mappingmtx.div(mappingmtx.sum(axis=1), axis=0)

    spaotsc_coord = np.zeros([mappingnorm.shape[0],2])
    spaotsc_coord[:,0] = np.matmul(mappingnorm.values,spcoor.coord_x.values)
    spaotsc_coord[:,1] = np.matmul(mappingnorm.values,spcoor.coord_y.values)
    return spaotsc_coord

def evaluation(mappingmtx,stcoor,scmetadata,stgtcelltype,celltypelist):
    sc2st = mappingmtx.idxmax(1)
    sc_mappedcoor =  stcoor.loc[sc2st.values,]
    sc_mappedcoor.index = sc2st.index
    
    pr_all=[]
    for i in range(stgtcelltype.shape[1]):
        scl =  mappingmtx.loc[scmetadata.celltype_mapped_refined==stgtcelltype.columns[i],].sum(0)
        pearson = scipy.stats.pearsonr(stgtcelltype.iloc[:,i],scl.values)[0]
        # pearson = scipy.stats.pearsonr(stgtcelltype.iloc[:,i],scl.values)[0]
        pr_all.append(pearson)
        if np.isnan(pearson):
            pearson=0
    pr_all = pd.DataFrame(pr_all,index=stgtcelltype.columns).T
    
    kl_loss=[]
    for i in range(stgtcelltype.shape[1]):
        scl =  mappingmtx.loc[scmetadata.celltype_mapped_refined==stgtcelltype.columns[i],].sum(0)
        klscore = scipy.special.kl_div(stgtcelltype.iloc[:,i]/stgtcelltype.iloc[:,i].sum(),scl.values/ scl.values.sum()+1e-15).sum()
        kl_loss.append(klscore)
    kl_loss = pd.DataFrame(kl_loss,index=stgtcelltype.columns).T

    
    
    scpred_coord = all_coord(mappingmtx,stcoor)
    sc2sc = scipy.spatial.distance.cdist(scpred_coord,scpred_coord)
        
    sc_gtcoord = scmetadata[['x_global','y_global']]
    true_sc2sc = scipy.spatial.distance.cdist(sc_gtcoord,sc_gtcoord)
    
    argmask = np.argsort(sc2sc,axis=1)
    true_argmask = np.argsort(true_sc2sc,axis=1)
    k_10 = []
    k_30 = []
    k_50 = []
    for i in range(argmask.shape[0]):
        k = 10
        k_10.append(len(set(argmask[i][:k].tolist()).intersection(set(true_argmask[i][:k].tolist()))))
        k = 30
        k_30.append(len(set(argmask[i][:k].tolist()).intersection(set(true_argmask[i][:k].tolist()))))
        k = 50
        k_50.append(len(set(argmask[i][:k].tolist()).intersection(set(true_argmask[i][:k].tolist()))))
    
    #spaotsc
    spaotsc_coord = k_coord(mappingmtx,stcoor, k=10)
    # tmp_coord = spaotsc_coord 
    dis_k10 = np.sqrt(((sc_gtcoord.values - spaotsc_coord)**2).sum(1))
    spaotsc_coord = k_coord(mappingmtx,stcoor, k=30)
    dis_k30 = np.sqrt(((sc_gtcoord.values - spaotsc_coord)**2).sum(1))
    spaotsc_coord = k_coord(mappingmtx,stcoor, k=50)
    dis_k50 = np.sqrt(((sc_gtcoord.values - spaotsc_coord)**2).sum(1))
    spaotsc_coord = all_coord(mappingmtx,stcoor)
    tmp_coord = spaotsc_coord
    dis_kall = np.sqrt(((sc_gtcoord.values - spaotsc_coord)**2).sum(1))
    

    return [pr_all,kl_loss, np.mean(k_10),np.mean(k_30),np.mean(k_50),dis_k10.mean(),dis_k30.mean(),dis_k50.mean(),dis_kall.mean(),dis_kall], tmp_coord, sc_gtcoord.values