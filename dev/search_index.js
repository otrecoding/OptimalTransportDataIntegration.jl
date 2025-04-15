var documenterSearchIndex = {"docs":
[{"location":"#OptimalTransportDataIntegration.jl","page":"Documentation","title":"OptimalTransportDataIntegration.jl","text":"","category":"section"},{"location":"","page":"Documentation","title":"Documentation","text":"CurrentModule = OptimalTransportDataIntegration","category":"page"},{"location":"","page":"Documentation","title":"Documentation","text":"","category":"page"},{"location":"","page":"Documentation","title":"Documentation","text":"Modules = [OptimalTransportDataIntegration]","category":"page"},{"location":"#OptimalTransportDataIntegration.Instance","page":"Documentation","title":"OptimalTransportDataIntegration.Instance","text":"struct Instance\n\nDefinition and initialization of an Instance structure\n\ndatafile : file name\ndistance : ∈ ( Cityblock, Euclidean, Hamming )\nindXA    : indexes of subjects of A with given X value\nindXB    : indexes of subjects of B with given X value\n\n\n\n\n\n","category":"type"},{"location":"#OptimalTransportDataIntegration.Solution","page":"Documentation","title":"OptimalTransportDataIntegration.Solution","text":"mutable struct Solution\n\ntsolve       : solution time\njointYZA     : joint distribution of Y and Z in A\njointYZB     : joint distribution of Y and Z in B\nestimatorZA  : estimator of probability of Z for individuals in base A\nestimatorYB  : estimator of probability of Y for individuals in base B\n\n\n\n\n\n","category":"type"},{"location":"#OptimalTransportDataIntegration.average_distance_to_closest-Tuple{Instance, Float64}","page":"Documentation","title":"OptimalTransportDataIntegration.average_distance_to_closest","text":"average_distance_to_closest(inst, percent_closest)\n\n\nCompute the cost between pairs of outcomes as the average distance between covariations of individuals with these outcomes, but considering only the percent closest neighbors\n\n\n\n\n\n","category":"method"},{"location":"#OptimalTransportDataIntegration.compute_pred_error!","page":"Documentation","title":"OptimalTransportDataIntegration.compute_pred_error!","text":"compute_pred_error!(sol, inst)\ncompute_pred_error!(sol, inst, proba_disp)\ncompute_pred_error!(sol, inst, proba_disp, mis_disp)\ncompute_pred_error!(\n    sol,\n    inst,\n    proba_disp,\n    mis_disp,\n    full_disp\n)\n\n\nCompute prediction errors in a solution\n\n\n\n\n\n","category":"function"},{"location":"#OptimalTransportDataIntegration.entropic_partial_wasserstein-NTuple{4, Any}","page":"Documentation","title":"OptimalTransportDataIntegration.entropic_partial_wasserstein","text":"entropic_partial_wasserstein(a, b, M, reg, m=nothing, numItermax=1000, \n    stopThr=1e-100, verbose=false, log=false)\n\nThis function is a Julia translation of the function entropic_partial_wasserstein in the Python Optimal Transport package. \n\nSolves the partial optimal transport problem and returns the OT plan\n\nThe function considers the following problem:\n\n    gamma = mathoparg min_gamma quad langle gamma\n             mathbfM rangle_F + mathrmreg cdotOmega(gamma)\n\n    st gamma mathbf1 leq mathbfa \n         gamma^T mathbf1 leq mathbfb \n         gamma geq 0 \n         mathbf1^T gamma^T mathbf1 = m\n         leq minmathbfa_1 mathbfb_1 \n\nwhere :\n\nmathbfM is the metric cost matrix\nOmega  is the entropic regularization term, Omega=sum_ij gamma_ijlog(gamma_ij)\nmathbfa and :math:\\mathbf{b} are the sample weights\nm is the amount of mass to be transported\n\nThe formulation of the problem has been proposed in :ref:[3] <references-entropic-partial-wasserstein> (prop. 5)\n\nParameters\n\na : Array (dima,) Unnormalized histogram of dimension `dima`\nb : Array (dimb,) Unnormalized histograms of dimension `dimb`\nM : Array (dima, dimb) cost matrix\nreg : Float Regularization term > 0\nm : Float, optional Amount of mass to be transported\nnumItermax : Int, optional Max number of iterations\nstopThr : Float, optional Stop threshold on error (>0)\nverbose : Bool, optional Print information along iterations\nlog : Bool, optional record log if True\n\nReturns\n\ngamma : (dima, dimb) ndarray   Optimal transportation matrix for the given parameters\nlog : Dict   log dictionary returned only if log is True\n\nExamples\n\njulia> a = [.1, .2];\n\njulia> b = [.1, .1];\n\njulia> M = [0. 1.; 2. 3.];\n\njulia> round.(entropic_partial_wasserstein(a, b, M, 1, m = 0.1), digits=2)\n2×2 Matrix{Float64}:\n 0.06  0.02\n 0.01  0.0\n\nReferences\n\n[3] Benamou, J. D., Carlier, G., Cuturi, M., Nenna, L., & Peyré, G.  (2015). Iterative Bregman projections for regularized transportation  problems. SIAM Journal on Scientific Computing, 37(2), A1111-A1138.\n\n\n\n\n\n","category":"method"},{"location":"#OptimalTransportDataIntegration.generate_data-Tuple{DataParameters}","page":"Documentation","title":"OptimalTransportDataIntegration.generate_data","text":"generate_data(params)\n\n\nFunction to generate data where X and (Y,Z) are categoricals\n\nthe function return a Dataframe with X1, X2, X3, Y, Z and the database id.\n\n\n\n\n\n","category":"method"},{"location":"#OptimalTransportDataIntegration.generate_xcat_ycat-Tuple{DataParameters}","page":"Documentation","title":"OptimalTransportDataIntegration.generate_xcat_ycat","text":"generate_xcat_ycat(params)\n\n\nFunction to generate data where X and (Y,Z) are categoricals\n\nthe function return a Dataframe with X1, X2, X3, Y, Z and the database id.\n\n\n\n\n\n","category":"method"},{"location":"#OptimalTransportDataIntegration.loss_crossentropy-Tuple{Any, Any}","page":"Documentation","title":"OptimalTransportDataIntegration.loss_crossentropy","text":"loss_crossentropy(Y, F)\n\nCross entropy is typically used as a loss in multi-class classification, in which case the labels y are given in a one-hot format. dims specifies the dimension (or the dimensions) containing the class probabilities. The prediction ŷ is usually probabilities but in our case it is also one hot encoded vector.\n\n\n\n\n\n","category":"method"},{"location":"#OptimalTransportDataIntegration.modality_cost-Tuple{Any, Any}","page":"Documentation","title":"OptimalTransportDataIntegration.modality_cost","text":"modality_cost(loss, weight)\n\nloss: matrix of size len(weight) * len(levels)\nweight: vector of weights \n\nReturns the scalar product <loss[level,],weight> \n\n\n\n\n\n","category":"method"},{"location":"#OptimalTransportDataIntegration.ot_joint","page":"Documentation","title":"OptimalTransportDataIntegration.ot_joint","text":"ot_joint(inst)\not_joint(inst, maxrelax)\not_joint(inst, maxrelax, lambda_reg)\not_joint(inst, maxrelax, lambda_reg, percent_closest)\not_joint(inst, maxrelax, lambda_reg, percent_closest, norme)\not_joint(\n    inst,\n    maxrelax,\n    lambda_reg,\n    percent_closest,\n    norme,\n    aggregate_tol\n)\not_joint(\n    inst,\n    maxrelax,\n    lambda_reg,\n    percent_closest,\n    norme,\n    aggregate_tol,\n    full_disp\n)\not_joint(\n    inst,\n    maxrelax,\n    lambda_reg,\n    percent_closest,\n    norme,\n    aggregate_tol,\n    full_disp,\n    solver_disp\n)\n\n\nModel where we directly compute the distribution of the outcomes for each individual or for sets of indviduals that similar values of covariates\n\naggregate_tol: quantify how much individuals' covariates must be close for aggregation\nreg_norm: norm1, norm2 or entropy depending on the type of regularization\npercent_closest: percent of closest neighbors taken into consideration in regularization\nlambda_reg: coefficient measuring the importance of the regularization term\nfull_disp: if true, write the transported value of each individual; otherwise, juste write the number of missed transports\nsolver_disp: if false, do not display the outputs of the solver\n\n\n\n\n\n","category":"function"},{"location":"#OptimalTransportDataIntegration.read_params-Tuple{AbstractString}","page":"Documentation","title":"OptimalTransportDataIntegration.read_params","text":"read_params(jsonfile)\n\n\nRead the data generation scenario from a JSON file\n\n\n\n\n\n","category":"method"},{"location":"#OptimalTransportDataIntegration.save_params-Tuple{AbstractString, DataParameters}","page":"Documentation","title":"OptimalTransportDataIntegration.save_params","text":"save_params(jsonfile, params)\n\n\nWrite the data generation scenario to a JSON file\n\n\n\n\n\n","category":"method"}]
}
