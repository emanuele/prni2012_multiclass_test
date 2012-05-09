"""Statistical tests for multiclass pattern discrimination.

Experiments on simulated data.
"""
import numpy as np
from scipy.special import gammaln
from scipy.stats import binom
from numpy.random import dirichlet
from experiments_PRNI2012 import log_multivariate_polya, logmean,\
     Delta, compute_logp_independent_block_mc, compute_logp_H_mc
from sklearn import svm, preprocessing

if __name__ == '__main__':

    np.random.seed(0)

    print __doc__
    
    # number of repetitions of the simulation
    reps = 150
    
    # number of iterations of MC integration
    iterations = 1e4
    
    # entries on the diagonal of the covariance matrix (sigmas)
    sigmas = [0.1, 0.5, 1]

    # test set size
    m = 100 # select a number with %4 = 0
    per_class = m/4

    # partitions of interest
    s_0 = [[0,1,2,3]]
    s_1 = [[0],[1],[2],[3]]
    s_s = [[0,1],[2,3]]
    s = [s_0,s_1,s_s]

    for sigma in sigmas:
        print '-------'
        print 'Sigma: %g' % sigma
        logp_H_mc_H_0 = []
        logp_H_mc_H_1 = []
        logp_H_mc_H_s = []
        
        BF_10 = []
        BF_s1 = []
        p_val = []
        for repetitions in range(reps):
            # create a datasset with 2 features and two classes - train set
            a = np.random.multivariate_normal([1,0],[[sigma,0],[0,sigma]],m/2)
            b = np.random.multivariate_normal([0,1],[[sigma,0],[0,sigma]],m/2)
            X1 = np.vstack((a,b))

            # create a datasset with 2 features and two classes - test set
            a = np.random.multivariate_normal([1,0],[[sigma,0],[0,sigma]],m/2)
            b = np.random.multivariate_normal([0,1],[[sigma,0],[0,sigma]],m/2)
            X2 = np.vstack((a,b))

            # create class labels
            y = np.array([0]*per_class+[1]*per_class+[2]*per_class+[3]*per_class)

            # scale train set
            scaler = preprocessing.Scaler().fit(X1)
            X1_s = scaler.transform(X1)

            # normalize train set
            normalizer = preprocessing.Normalizer().fit(X1_s)
            X1_s_n = normalizer.transform(X1_s)

            # scale and normalize test set
            X2_s = scaler.transform(X2)
            X2_s_n = normalizer.transform(X2_s)
            
            # select the classifier
            clf = svm.LinearSVC()
    
            # fit classifier to pre-processed train set
            clf.fit(X1_s_n, y)

            # predict pre-processed test set
            results = clf.predict(X2_s_n)

            # number of classes
            c = len(np.unique(y))

            # create confusion matrix
            N = np.zeros((c,c))
            for i in range(c):
                for j in range(c):
                    N[i,j] = np.mean(np.all([np.array(y == j),np.array(results == i)],axis=0))*len(results)
            #print N

            # create non informative flat prior
            alpha = np.ones(N.shape)

            posteriors = []
            logp_H_mc_H_0.append(compute_logp_H_mc(N, alpha, s[0], iterations=iterations))
            logp_H_mc_H_1.append(compute_logp_H_mc(N, alpha, s[1], iterations=iterations))
            logp_H_mc_H_s.append(compute_logp_H_mc(N, alpha, s[2], iterations=iterations))
            #print cm

            # do binomial test
            binomN = binom(N.sum(), 1./c)
            p_val.append(binomN.sf(np.trace(N)))

            # calculate Bayes factors
            BF_10.append(np.exp(np.array(logp_H_mc_H_1[-1])-np.array(logp_H_mc_H_0[-1])))
            BF_s1.append(np.exp(np.array(logp_H_mc_H_s[-1])-np.array(logp_H_mc_H_1[-1])))

        # print average results
        print
        print 'average %i repetitions - p-value: %.5g' % (reps,np.mean(p_val))
        print 'average %i repetitions - BF_10: %.5g' % (reps, np.mean(BF_10))
        print 'average %i repetitions - BF_s1: %.5g (+- %.3g)' % (reps,np.mean(BF_s1),np.std(BF_s1)/reps)
        print "\t (Monte Carlo: %g iterations)" % iterations
        print '-------'

