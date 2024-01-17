#uLSIF python version
import numpy as np
np.random.seed(0)

def uLSIF(x_de,x_nu,sigma_list=np.logspace(-3, 1, 9),lambda_list=np.logspace(-3, 1, 9),b=100,fold=0):
#
# Unconstrained least-squares importance fitting (with leave-one-out cross validation)
#
# Estimating ratio of probability densities
#   \frac{ p_{nu}(x) }{ p_{de}(x) }
# from samples
#    { xde_i | xde_i\in R^{d} }_{i=1}^{n_{de}} 
# drawn independently from p_{de}(x) and samples
#    { xnu_i | xnu_i\in R^{d} }_{i=1}^{n_{nu}} 
# drawn independently from p_{nu}(x).
#
# Usage:
#       [wh_x_de,wh_x_re]=uLSIF(x_de,x_nu,x_re,sigma_list,lambda_list,b)
#
# Input:
#    x_de:         d by n_de sample matrix corresponding to `denominator' (iid from density p_de)
#    x_nu:         d by n_nu sample matrix corresponding to `numerator'   (iid from density p_nu)
#    x_re:         (OPTIONAL) d by n_re reference sample matrix
#    sigma_list:   (OPTIONAL) Gaussian width
#                  If sigma_list is a vector, one of them is selected by cross validation.
#                  If sigma_list is a scalar, this value is used without cross validation.
#                  If sigma_list is empty/undefined, Gaussian width is chosen from
#                  some default canditate list by cross validation.
#    lambda_list: (OPTIONAL) regularization parameter
#                 If lambda_list is a vector, one of them is selected by cross validation.
#                 If lambda_list is a scalar, this value is used without cross validation
#                 If lambda_list is empty, Gaussian width is chosen from
#                 some default canditate list by cross validation
#    b:           (OPTINLAL) positive integer representing the number of kernels (default: 100)
#    fold:        (OPTINLAL) positive integer representing the number of folds
#                 in cross validation / 0: leave-one-out (default: 0)
#
# Output:
#    wh_x_de:     estimates of density ratio w=p_nu/p_de at x_de
#    wh_x_re:     estimates of density ratio w=p_nu/p_de at x_re (if x_re is provided)
#
# (c) Masashi Sugiyama, Department of Compter Science, Tokyo Institute of Technology, Japan.
#     sugi@cs.titech.ac.jp,     http://sugiyama-www.cs.titech.ac.jp/~sugi/software/uLSIF/
    d, n_de = np.shape(x_de)
    d_nu, n_nu = np.shape(x_nu)
    if d != d_nu:
        return print(ValueError('Dimension of two samples are different!!!')) 
    rand_index = np.random.permutation(n_nu)  
    b = min(b,n_nu)
    if n_nu < n_de:
        x_ce = x_nu[:,rand_index[:b]] 
    else:
        x_ce = x_nu[:,rand_index[:b]] 
    n_min = min(n_de,n_nu)
    x_de2 = np.sum(np.power(x_de,2), axis=0)
    x_nu2 = np.sum(np.power(x_nu,2), axis=0)
    x_ce2 = np.sum(np.power(x_ce,2), axis=0)


    dist2_x_de = np.tile(x_ce2, (n_de,1)).T + np.tile(x_de2, (b, 1)) - 2 * np.dot(x_ce.T, x_de)
    dist2_x_nu = np.tile(x_ce2, (n_nu,1)).T + np.tile(x_nu2, (b, 1)) - 2 * np.dot(x_ce.T, x_nu)    
    score_cv = np.zeros((len(sigma_list),len(lambda_list)))

    if len(sigma_list)==1 and len(lambda_list)==1:
        sigma_chosen = sigma_list
        lambda_chosen = lambda_list
    else:
        if fold != 0:
            cv_index_nu = np.random.permutation(n_nu)
            cv_split_nu = np.floor(np.arange(0, n_nu) * fold / n_nu) + 1
            cv_index_de = np.random.permutation(n_de)
            cv_split_de = np.floor(np.arange(0, n_de) * fold / n_de) + 1
            

        for sigma_index in range(len(sigma_list)):
            sigma = sigma_list[sigma_index]
            K_de = np.exp(-dist2_x_de / (2 * sigma**2))
            K_nu = np.exp(-dist2_x_nu / (2 * sigma**2))
            if fold == 0:    # leave-one-out cross-validation
                K_de2 = K_de[:, :n_min]
                K_nu2 = K_nu[:, :n_min]
                H = np.dot(K_de, K_de.T) / K_de.shape[1]
                h = np.mean(K_nu, axis=1)   
            for lambda_index in range(len(lambda_list)):
                lambda_ = lambda_list[lambda_index]
                if fold==0:  # leave-one-out cross-validation
                    C = H + lambda_ * (n_de - 1) / n_de * np.eye(b)
                    invC = np.linalg.inv(C)
                    beta = np.dot(invC, h)
                    invCK_de = np.dot(invC, K_de2)
                    tmp = n_de * np.ones(n_min) - np.sum(K_de2 * invCK_de, axis=0)
                    B0 = np.dot(beta[:, np.newaxis], np.ones((1, n_min))) + np.dot(invCK_de, np.diag(np.dot(beta.T, K_de2) / tmp))
                    B1 = np.dot(invC, K_nu2) + np.dot(invCK_de, np.diag(np.sum(K_nu2 * invCK_de, axis=0) / tmp))
                    A = np.maximum(0, (n_de - 1) / (n_de * (n_nu - 1)) * (n_nu * B0 - B1))
                    wh_x_de2 = np.sum(K_de2 * A, axis=0).T
                    wh_x_nu2 = np.sum(K_nu2 * A, axis=0).T
                    score_cv[sigma_index, lambda_index] = np.mean(np.power(wh_x_de2,2)) / 2 - np.mean(wh_x_nu2) 
                else: # k-fold cross-validation  
                    score_tmp = np.zeros(fold)
                    for k in range(1, fold + 1):
                        Ktmp = K_de[:, cv_index_de[cv_split_de != k]]
                        alphat_cv = np.linalg.solve(np.dot(Ktmp,Ktmp.T) / Ktmp.shape[1] + lambda_ * np.eye(b),
                                                    np.mean(K_nu[:, cv_index_nu[cv_split_nu != k]], axis=1))
                        alphah_cv = np.maximum(0, alphat_cv)
                        score_tmp[k - 1] = np.mean((K_de[:, cv_index_de[cv_split_de == k]].T.dot(alphah_cv))**2) / 2 \
                                        - np.mean(K_nu[:, cv_index_nu[cv_split_nu == k]].T.dot(alphah_cv))

                    score_cv[sigma_index, lambda_index] = np.mean(score_tmp)
        score_cv_tmp = np.min(score_cv, axis=1)
        lambda_chosen_index = np.argmin(score_cv, axis=1)
        score = np.min(score_cv_tmp)
        sigma_chosen_index = np.argmin(score_cv_tmp)
        lambda_chosen = lambda_list[lambda_chosen_index[sigma_chosen_index]]
        sigma_chosen = sigma_list[sigma_chosen_index]

    print(f'sigma = {sigma_chosen}, lambda = {lambda_chosen}')
    K_de = np.exp(-dist2_x_de / (2 * sigma_chosen**2)) 
    K_nu = np.exp(-dist2_x_nu / (2 * sigma_chosen**2))
    A_t = np.dot(K_de, K_de.T) / n_de + lambda_chosen * np.eye(b)
    alphat = np.linalg.solve(A_t, np.mean(K_nu, axis=1))
    alphah = np.maximum(0, alphat)
    wh_x_de = np.dot(alphah.T, K_de)

    
    # d, n_re = np.shape(x_re)
    # x_re2 = np.sum(np.power(x_re,2), axis=0)
    # dist2_x_re = np.tile(x_ce2, (n_re,1)).T + np.tile(x_re2, (b, 1)) - 2 * np.dot(x_ce.T, x_re)
    # wh_x_re = np.dot(alphah.T, np.exp(-dist2_x_re / (2 * sigma_chosen**2)))
    # return wh_x_de, wh_x_re    
    return wh_x_de

