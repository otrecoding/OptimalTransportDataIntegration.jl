function proxim_dist(data, prox=0.30)

    nbcvar = length(names(data)) - 3

    base = data.database

    indA = findall(base .== 1)
    indB = findall(base .== 2)

    nA = length(indA)
    nB = length(indB)

    Xobserv = dat.iloc[:, 3 : dat.shape[1]].values
    Yobserv = data.Y
    Zobserv = data.Z
    Xobserv = np.r_[Xobserv[indA], Xobserv[indB]]
    Yobserv = np.r_[Yobserv[indA], Yobserv[indB]]
    Zobserv = np.r_[Zobserv[indA], Zobserv[indB]]

    indA = range(nA)
    indB = range(nA, nA + nB)

    # print(np.max(Yobserv[indA]))
    # Modify Y and Z so that they go from 1 to the number of modalities
    Y = np.arange(np.max(Yobserv[indA]))
    Z = np.arange(np.max(Zobserv[indB]))

    # list the distinct modalities in A and B
    indY = list()
    indZ = list()

    for m in Y:
        indY.append(np.where(Yobserv[0:nA] == (m+1))[0])
    for m in Z:
        indZ.append(np.where(Zobserv[nA : (nA + nB)] == (m+1))[0])

    a = Xobserv[indA, :]
    b = Xobserv[indB, :]

    if norm == "E":

        D = cdist(a, b, metric="sqeuclidean")
        DA = cdist(a, a, metric="sqeuclidean")
        DB = cdist(b, b, metric="sqeuclidean")

    elif norm == "H":

        D = cdist(a, b, metric="hamming")
        DA = cdist(a, a, metric="hamming")
        DB = cdist(b, b, metric="hamming")

    # aggregate both bases
      # Compute the indexes of individuals with same covariates
    A = range(nA)
    B = range(nB)

    Xval  = np.unique(Xobserv, axis=0)


    indXA = list()
    indXB = list()

    n_Xval = Xval.shape[0]
    indXAA = np.zeros((n_Xval, nA))
    indXBB = np.zeros((n_Xval, nB))

    # aggregate both bases

    indXA = list()
    indXB = list()

    n_Xval = Xval.shape[0]
    indXAA = np.zeros((n_Xval, nA))
    indXBB = np.zeros((n_Xval, nB))

    for x in range(n_Xval):

        Xobs_A = Xobserv[:nA]
        Xobs_B = Xobserv[nA:]

        if norm == "E":

            distA = cdist(Xval[x][np.newaxis, :], Xobs_A, metric="euclidean").ravel()
            distB = cdist(Xval[x][np.newaxis, :], Xobs_B, metric="euclidean").ravel()

        elif norm == "H":

            distA = cdist(Xval[x][np.newaxis, :], Xobs_A, metric="hamming").ravel()#[np.newaxis, :]
            distB = cdist(Xval[x][np.newaxis, :], Xobs_B, metric="hamming").ravel()


        indXA.append(np.where(distA < prox * np.max(distA))[0])
        indXB.append(np.where(distB < prox * np.max(distB))[0])

        indXAA[x, :] = distA
        indXBB[x, :] = distB

    if prox == 0:
        indXA = list()
        indXB = list()
        

        for x in range(n_Xval):
    
                indXA.append(np.zeros(nA))
                indXB.append(np.zeros(nB))
        bbb = np.argmin(indXAA, axis=0)
        for   x in range(n_Xval):
            indXA[x]=(np.where(bbb == x)[0])

        ccc = np.argmin(indXBB, axis=0)
        for   x in range(n_Xval):
            indXB[x]=(np.where(ccc == x)[0])

    return dict(
        nA=nA,
        nB=nB,
        Xobserv=Xobserv,
        profile=np.unique(Xobserv, axis=1),
        Yobserv=Yobserv,
        Zobserv=Zobserv,
        D=D,
        Y=Y,
        Z=Z,
        indY=indY,
        indZ=indZ,
        indXA=indXA,
        indXB=indXB,
        DA=DA,
        DB=DB,
    )


end
