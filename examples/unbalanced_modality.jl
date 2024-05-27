# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Julia 1.10.3
#     language: julia
#     name: julia-1.10
# ---

import Pkg; Pkg.add(url="https://github.com/otrecoding/OTRecod.jl.git")

# +
using OptimalTransportDataIntegration
using OTRecod
using CSV
using DataFrames

params = DataParameters(nA = 1000, nB = 500)

#data = generate_xcat_ycat(params)
data = CSV.read("data.csv", DataFrame)
# -

onecold(X) = map(argmax, eachrow(X))

Xnames_hot, X_hot, Y, Z, XA_hot, YA, XB_hot, ZB, YB_true, ZA_true = prep_data(data)
Xnames_hot

# jdonnées individuelles annexées par i

XA_hot

XA_hot_i = XA_hot
XB_hot_i  = XB_hot
yA_i  = onecold(YA)
zB_i  = onecold(ZB)

nA_i, nB_i  = size(XA_hot_i, 1), size(XB_hot_i, 1)

XYA_i = hcat(XA_hot_i, yA_i)
XZB_i = hcat(XB_hot_i, zB_i)

# +
import Distances: Hamming

X = Matrix(data[!, ["X1", "X2", "X3"]])
Y = Vector(data.Y)
Z = Vector(data.Z)
database = data.database

dist_choice = Hamming()
    
instance = OTRecod.Instance( database, X, Y, Z, dist_choice)
    
# -

# # Compute data for aggregation of the individuals

estim_XA_YA = Dict([
        ((x, y), length(indXA[x][findall(Yobserv[indXA[x]] .== y)]) / nA) for x = 1:nbX,
        y in Y
    ])
estim_XB_ZB = Dict([
        ((x, z), length(indXB[x][findall(Zobserv[indXB[x].+nA] .== z)]) / nB) for
        x = 1:nbX, z in Z
    ])       

wa = collect(values(estim_XA_YA))
wb = collect(values(estim_XB_ZB))        

Xvalues = sort(unique(eachrow(one_hot_encoder(X))))
Yobserv = sort(unique(instance.Yobserv))
Zobserv = sort(unique(instance.Zobserv))

wa2 = wa[wa .> 0]
wb2 = wb[wb .> 0]


# +
import .Iterators: flatten, product

XYA = stack([[x...,y] for (x,y) in product(Xobserv,Yobserv)], dims=1)
XYB = stack([[x...,y] for (x,y) in product(Xobserv,Zobserv)], dims=1)

# + endofcell="---"
XYA2 = XYA[wa !=0,:] ### XYA observés
XZB2 = XZB[wb !=0,:] ### XZB observés

Y_hot = onehot(Y)
Z_hot = onehot(Z)

nx= X.shape[1] ## Nb modalités x 

XA_hot = XYA2[:,0:nx] # les x parmi les XYA observés, potentiellement des valeurs repetées 
XB_hot = XZB2[:,0:nx] # les x dans XZB observés, potentiellement des valeurs repetées 
yA = XYA2[:,nx]  ## les y  parmi les XYA observés, des valeurs repetées 
zB = XZB2[:,nx] # les z dans XZB observés, potentiellement des valeurs repetées 
yA_hot = onehot(yA)
zB_hot = onehot(zB)
# -
# --

# # +
def OptimalModality(Values,Loss,Weight):
    """
    Values: vector of possible values
    Weight: vector of weights 
    Loss: matrix of size len(Weight) * len(Values)
    Returns an argmin over value in Values of the scalar product <Loss[value,],Weight> 
    """
    
    CostForEachModality=[]
    for j in range(0,len(Values)):
        s=0
        for i in range(0,len(Loss)):
            s+=Loss[i,j]*Weight[i]
        CostForEachModality.append(s)
    return(Values[np.argmin(CostForEachModality)])

        
# -

# ## Algorithm

# # +
# Initialisation 
nA = XYA2.shape[0] # number of observed different values in A
nB = XZB2.shape[0] # number of observed different values in B
nbrvarX = 3
C0 = cdist(XA_hot, XB_hot, metric="hamming") * nx / nbrvarX
C = C0 / np.max(C0)
dimXZB = XZB2.shape[1]
dimXYA = XYA2.shape[1]


NumberOfIterations=3

Y =range(yA_hot.shape[1])
Z =range(zB_hot.shape[1])
Y_hot = onehot(Y)
Z_hot = onehot(Z)
yB_pred = np.zeros((nB))
zA_pred = np.zeros((nA))

Y_loss = loss_crossentropy(yA_hot, Y_hot)
Z_loss= loss_crossentropy(zB_hot, Z_hot) # Zc=onehot(possible values of Z),zB2 onehot(zB), où zB est les z dans XZB observés, potentiellement des valeurs repetées 

for iter in range(NumberOfIterations):
    ### Optimal Transport
    G = ot.unbalanced.mm_unbalanced(wa2, wb2, C,reg_m=0.1, div='kl') #unbalanced
    # G =ot.unbalanced.mm_unbalanced2(wa2, wb2, C, reg_m=0.1, c=None, reg=0, div='kl')#unbalanced
    #G = ot.emd(ot.unif(nA), ot.unif(nB), C) #balanced
    
    #if method == "sinkhorn":
           #G = ot.sinkhorn(ot.unif(nA), ot.unif(nB), C, reg)
    #if method == "emd":
            #G = ot.emd(ot.unif(nA), ot.unif(nB), C) #balanced
    ### Compute best f:XxZ->Y

    for j in range(nB):
        yB_pred[j]=OptimalModality(Y,Y_loss,G[:,j])
    yB_pred_hot = onehot(yB_pred)
    
    ### Compute best g: XxY-->Z
  
    for i in range(nA):
        zA_pred[i] = OptimalModality(Z,Z_loss,G[i,:])

    zA_pred_hot = onehot(zA_pred)

    ### Update Cost matrix
    alpha1=1/np.max(loss_crossentropy(yA_hot, yB_pred_hot))
    alpha2=1/np.max(loss_crossentropy(zB_hot, zA_pred_hot))

    chinge1 = alpha1 * loss_crossentropy(yA_hot, yB_pred_hot)
    chinge2 = alpha2 * loss_crossentropy(zB_hot, zA_pred_hot).T
    fcost = chinge1 + chinge2
  
    C = C0 / np.max(C0) + fcost

    #### Predict
    zA_pred_hot_i=np.zeros((nA_i,len(Z)))
    for i in range(nA_i):
         ind = np.where((XYA_i[i,:] == XYA2).all(axis=1))[0][0]
         zA_pred_hot_i[i,:] = zA_pred_hot[ind,:]
    yB_pred_hot_i=np.zeros((nB_i,len(Y)))
    for i in range(nB_i):
         ind = np.where((XZB_i[i,:] == XZB2).all(axis=1))[0][0]
         yB_pred_hot_i[i,:] = yB_pred_hot[ind,:]
 

    YB_pred= np.argmax(yB_pred_hot_i, 1) + 1
    ZA_pred = np.argmax(zA_pred_hot_i, 1) + 1


    ### Evaluate 

    est = (np.sum(YB_true == YB_pred) + np.sum(ZA_true == ZA_pred)) / (nA_i + nB_i)
    print(est)
    print(np.sum(YB_true == YB_pred)/nB_i)
    print(np.sum(ZA_true == ZA_pred)/nA_i)

# -

# ## Monte Carlo Simulations

params = DataParameters(nA=1000, nB=1000, mB=[2,0,0], eps=0, p=0.2)

# # +
MC_simulations=100
NumberOfIterations=3 # number of iterations for the BDC used on each data
reg_m=0.01 # marginal regularization parameter of unbalanced OT
PredictionQuality=np.zeros(MC_simulations)

for simulation in range(MC_simulations):
    
    ## Generate new data  

    csv_file = generator_xcat_ycat(params)
    df = csv_file
    df["X1_1"] = onehot(df.X1)[:, 1]
    df["X2_1"] = onehot(df.X2)[:, 1]
    df["X2_2"] = onehot(df.X2)[:, 2]
    df["X3_1"] = onehot(df.X3)[:, 1]
    df["X3_2"] = onehot(df.X3)[:, 2]
    df["X3_3"] = onehot(df.X3)[:, 3]
    database = df.database.values

    dba = df[database == 1]
    dbb = df[database == 2]

    YBtrue = dbb.Y.values
    ZAtrue = dba.Z.values

    Xnames = ["X1_1", "X2_1", "X2_2", "X3_1", "X3_2", "X3_3"]

    X = df[Xnames].values
    Y = df.Y.values
    Z = df.Z.values

    XA = dba[Xnames].values
    XB = dbb[Xnames].values

    yA = onehot(dba.Y)
    zB = onehot(dbb.Z)
    ZA = onehot(dba.Z)
    YB = onehot(dbb.Y)

    Xnames, X, Y, Z, XA, YA, XB, ZB, YBtrue, ZAtrue = prep_data_valerie(df)

    df = pd.DataFrame(np.c_[database, Y, Z, X],
                    columns=["database", "Y", "Z", *Xnames])
    percent_knn=1
    dist_choice="H"
    prox_X = 1
    prepa = OT_prepa(df, dist_choice, percent_knn, prox_X) ## a-t-on besoin de ça ? ça affiche lamba_reg=1


    ### jdonnées individuelles annexées par i
    XAi = XA
    XBi = XB
    yAi = dba.Y
    zBi = dbb.Z

    nAi = XAi.shape[0]
    nBi = XBi.shape[0]

    XYAi = np.c_[XAi, yAi]
    XZBi = np.c_[XBi, zBi]


    estim_XA_YA = prepa["estim_XA_YA"] ##mettre en vecteur Xoberv en ligne Y observ en colonne
    estim_XB_ZB = prepa["estim_XB_ZB"] ##mettre en vecteur

    X = prepa["Xvalues"]
    Y = prepa["Y"]
    Z = prepa["Z"]
    wa = np.ravel(estim_XA_YA)
    wb = np.ravel(estim_XB_ZB)



    Xobserv = np.unique(prepa["Xobserv"], axis=0)
    Yobserv = np.unique(prepa["Yobserv"], axis=0)
    Zobserv = np.unique(prepa["Zobserv"], axis=0)



    wa2 = wa[wa !=0]
    wb2 = wb[wb !=0]
   
    XYA = pd.DataFrame([(*x,y) for x,y in product(Xobserv,Yobserv)]).values
    XZB = pd.DataFrame([(*x,y) for x,y in product(Xobserv,Zobserv)]).values


    XYA2 = XYA[wa !=0,:] ### XYA observés

    XZB2 = XZB[wb !=0,:] ### XZB observés

    Yc = onehot(Y)
    Zc = onehot(Z)

    nx= X.shape[1] ## Nb modalités x 

    XA = XYA2[:,0:nx] # les x parmi les XYA observés, potentiellement des valeurs repetées 
    XB = XZB2[:,0:nx] # les x dans XZB observés, potentiellement des valeurs repetées 
    yA = XYA2[:,nx]  ## les y  parmi les XYA observés, des valeurs repetées 
    zB = XZB2[:,nx] # les z dans XZB observés, potentiellement des valeurs repetées 
    yA2 = onehot(yA)
    zB2 = onehot(zB)


    ### Initialisation of different variables :

    nA = XYA2.shape[0] # number of observed different values in A
    nB = XZB2.shape[0] # number of observed different values in B
    nbrvarX = 3
    C0 = cdist(XA, XB, metric="hamming") * nx / nbrvarX
    C = C0 / np.max(C0)
    dimXZB = XZB2.shape[1]
    dimXYA = XYA2.shape[1]



    Y =range(yA2.shape[1])
    Z =range(zB2.shape[1])
    Yc = onehot(Y)
    Zc = onehot(Z)
    yBpred = np.zeros((nB))
    zApred = np.zeros((nA))

    Y_loss = loss_crossentropy(yA2, Yc)
    Z_loss= loss_crossentropy(zB2, Zc) # Zc=onehot(possible values of Z),zB2 onehot(zB), où zB est les z dans XZB observés, potentiellement des valeurs repetées 



    ### Algorithm
    for iter in range(NumberOfIterations):
        ### Optimal Transport
        G = ot.unbalanced.mm_unbalanced(wa2, wb2, C,reg_m=reg_m, div='kl')

        ### Compute best f:XxZ->Y

        for j in range(nB):
            yBpred[j]=OptimalModality(Y,Y_loss,G[:,j])
        yBpred2 = onehot(yBpred)
        
        ### Compute best g: XxY-->Z
    
        for i in range(nA):
            zApred[i] = OptimalModality(Z,Z_loss,G[i,:])

        zApred2 = onehot(zApred)

        ### Update Cost matrix
        print(yA2.shape, yBpred2.shape)
        alpha1=1/np.max(loss_crossentropy(yA2, yBpred2))
        print(zB2.shape, zApred2.shape)
        alpha2=1/np.max(loss_crossentropy(zB2, zApred2))

        chinge1 = alpha1 * loss_crossentropy(yA2, yBpred2)
        chinge2 = alpha2 * loss_crossentropy(zB2, zApred2).T
        fcost = chinge1 + chinge2
    
        C = C0 / np.max(C0) + fcost

        #### Predict
        zApredi=np.zeros((nAi,len(Z)))
        for i in range(nAi):
            ind = np.where((XYAi[i,:] == XYA2).all(axis=1))[0][0]
            zApredi[i,:] = zApred2[ind,:]
        yBpredi=np.zeros((nBi,len(Y)))
        for i in range(nBi):
            ind = np.where((XZBi[i,:] == XZB2).all(axis=1))[0][0]
            yBpredi[i,:] = yBpred2[ind,:]
    

        YBpred= np.argmax(yBpredi, 1) + 1
        ZApred = np.argmax(zApredi, 1) + 1


        ### Evaluate 

        est = (np.sum(YBtrue == YBpred) + np.sum(ZAtrue == ZApred)) / (nAi + nBi)

    PredictionQuality[simulation]=est
# -

# ## Notes on the results :
# - the algorithm performs better for reg_m=0.01 than reg_m=0.1 (marginal regularization parameter of Unbalanced OT)
# - ce qui est bizare c'est que c'est pour des valeurs plus grands 0.1 ou 0.05 que j'ai eu une erreur sur la forme de y ou z (car une valeur n'a jamais etait predite et donc  le resultat de onehot(yBpred) n'a pas la bonne dimension ). Normalement plus ce param est grand, plus la marginale devrait coller aux empiriques observés.
# - for reg_m<=0.01 the performance seems to stay the same

PredictionQuality

np.mean(PredictionQuality)

# ## Compare with simple learning 

# # +
MC_simulations=100
SimplePredictionQuality=np.zeros(MC_simulations)
params = DataParameters(nA=1000, nB=1000, mB=[2,0,0], eps=0, p=0.2)
method = "emd"
nb_epoch = 500
batch_size=1000

for simulation in range(MC_simulations):

    

    csv_file = generator_xcat_ycat(params)
    df = csv_file
    df["X1_1"] = onehot(df.X1)[:, 1]
    df["X2_1"] = onehot(df.X2)[:, 1]
    df["X2_2"] = onehot(df.X2)[:, 2]
    df["X3_1"] = onehot(df.X3)[:, 1]
    df["X3_2"] = onehot(df.X3)[:, 2]
    df["X3_3"] = onehot(df.X3)[:, 3]
    database = df.database.values

    dba = df[database == 1]
    dbb = df[database == 2]

    YBtrue = dbb.Y.values
    ZAtrue = dba.Z.values

    Xnames = ["X1_1", "X2_1", "X2_2", "X3_1", "X3_2", "X3_3"]

    X = df[Xnames].values
    Y = df.Y.values
    Z = df.Z.values

    XA = dba[Xnames].values
    XB = dbb[Xnames].values

    yA = onehot(dba.Y)
    zB = onehot(dbb.Z)
    ZA = onehot(dba.Z)
    YB = onehot(dbb.Y)

    Xnames, X, Y, Z, XA, YA, XB, ZB, YBtrue, ZAtrue = prep_data_valerie(df)

    df = pd.DataFrame(np.c_[database, Y, Z, X],
                    columns=["database", "Y", "Z", *Xnames])
    
    percent_knn=1
    dist_choice="H"
    prox_X = 1
    prepa = OT_prepa(df, dist_choice, percent_knn, prox_X)

    dimX = XA.shape[1]
    dimY = yA.shape[1]
    dimZ = zB.shape[1]

    # Initializations
    nA = XA.shape[0]
    nB = XB.shape[0]
    # classifiers
    get_model = get_model_en
    g = get_model(dimX, dimY)
    f = get_model(dimX, dimZ)

    # Init initial g(.)
    g.fit(XA, yA, epochs=nb_epoch, batch_size=batch_size, verbose=0)
    f.fit(XB, zB, epochs=nb_epoch, batch_size=batch_size, verbose=0)
    yBpred = g.predict(XB)
    zApred = f.predict(XA)
    YBpred = onecold(yBpred)
    ZApred = onecold(zApred)  
    #estimation de l'accuracy

    est = (np.sum(YBtrue == YBpred) + np.sum(ZAtrue == ZApred)) / (nA + nB)
    SimplePredictionQuality[simulation]=est
    
# -

# C'est beaucoup plus long que l'autre methode
#
# To test with other parameters: ex less epochs

np.mean(SimplePredictionQuality)

# # +
Y = range(4)
Z = range(3)

Y_hot = onehot(Y)
Z_hot = onehot(Z)
# -

print(Y_hot)
print(Z_hot)

np.max(loss_crossentropy(Y_hot, Y_hot))

np.max(loss_crossentropy(Z_hot,Z_hot))
# ---


