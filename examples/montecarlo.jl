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

# ## Monte Carlo Simulations
# +
using OptimalTransportDataIntegration
using OTRecod
using CSV
using DataFrames
import PythonOT
import .Iterators: product
import Distances: pairwise

params = DataParameters(nA=1000, nB=1000, mB=[2,0,0], eps=0, p=0.2)

#data = generate_xcat_ycat(params)
data = CSV.read("data.csv", DataFrame)
dba = subset(data, :database => ByRow(==(1)))
dbb = subset(data, :database => ByRow(==(2)))
# -

Xhot = one_hot_encoder(Matrix(data[!, ["X1", "X2", "X3"]]))


# +
MC_simulations=100
NumberOfIterations=3 # number of iterations for the BDC used on each data
reg_m = 0.01 # marginal regularization parameter of unbalanced OT
PredictionQuality=Float64[]

#for simulation in 1:MC_simulations
# +
YBtrue = dbb.Y
ZAtrue = dba.Z

X = one_hot_encoder(Matrix(data[!, ["X1", "X2", "X3"]]))
Y = data.Y
Z = data.Z

XA = one_hot_encoder(Matrix(dba[!, ["X1", "X2", "X3"]]))
XB = one_hot_encoder(Matrix(dbb[!, ["X1", "X2", "X3"]]))

yA = one_hot_encoder(dba.Y)
zB = one_hot_encoder(dbb.Z)
ZA = one_hot_encoder(dba.Z)
YB = one_hot_encoder(dbb.Y)
# +
Xnames, X, Y, Z, XA, YA, XB, ZB, YBtrue, ZAtrue = prep_data(data)
# + endofcell="--"
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
# --


