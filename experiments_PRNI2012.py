"""Statistical tests for multiclass pattern discrimination.

Confusion matrices from ICANN2011 MEG competition.
See http://www.cis.hut.fi/icann2011/meg/megicann_proceedings.pdf
p.14

New BSD License

Copyright (c) 2012, Emanuele Olivetti
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


import numpy as np
from scipy.special import gammaln
from scipy.stats import binom
from numpy.random import dirichlet


def log_multivariate_polya(N, alpha):
    """Multivariate Polya log PDF. Vectorized and stable implementation.
    """
    
    N = np.atleast_1d(N)
    alpha = np.atleast_1d(alpha)
    assert(N.size==alpha.size)
    m = N.sum()
    A = alpha.sum()
    log_likelihood = gammaln(m+1) - gammaln(N+1).sum() # log(\frac{N!}{\prod_i (N_i)!})
    log_likelihood += gammaln(A) - gammaln(alpha).sum() # log(\frac{\Gamma(\sum_i alpha_i)}{\prod_i(\Gamma(\alpha_i))})
    log_likelihood += gammaln(N + alpha).sum() - gammaln(m + A) # log(\frac{\prod_i(\Gamma(N_i +\alpha_i))}{\Gamma(\sum_i N_i+\alpha_i)})
    return log_likelihood


def logmean(loga):
    """Log of the mean of a sample of values (loga) in logscale.
    """
    return reduce(np.logaddexp, loga) - np.log(loga.size)


def Delta(A, s):
    """Given a squared matrix A (e.g. a confusion matrix) and a
    partition s of the rows/columns indices (e.g. of the classes), it
    returns a vector whose elements are the aggregated values of each
    submatrix of A defined by s concatenated with all elements of A
    not in the submatrices defined through s.
    """
    if len(s[0]) == A.shape[1]: return np.array([A.sum()])
    def no(group):
        return filter(lambda a: a not in group, range(A.shape[1]))
    A_blocks = [A[np.ix_(group,group)].sum() for group in s]
    A_free = np.concatenate([A[np.ix_(group,no(group))].flatten() for group in s]).tolist()
    return np.array(A_blocks + A_free)


def compute_logp_independent_block_mc(N, alpha_row=None, alpha_col=None, iterations=1e5):
    """Compute the montecarlo log likelihood of a matrix under the
    assumption of independence.
    """
    if N.size == 1 : return 0
    if alpha_row is None: alpha_row = np.ones(N.shape[1])
    if alpha_col is None: alpha_col = np.ones(N.shape[0])
    theta_row = dirichlet(alpha_row, size=int(iterations)).T
    theta_col = dirichlet(alpha_col, size=int(iterations)).T
    Theta = theta_row[:,None,:] * theta_col
    logp_ibs = gammaln(N.sum()+1) - gammaln(N+1).sum() + (np.log(Theta)*N[:,:,None]).sum(0).sum(0)
    return logmean(logp_ibs)


def compute_logp_H_mc(N, alpha, s, iterations=1e5):
    """Compute the partly analytical and partly montecarlo log
    likelihood of the confusion matrix N with hyper-prior alpha (in a
    multivariate-Dirichlet sense) according to a partition s.
    """
    logp_H = log_multivariate_polya(Delta(N, s), Delta(alpha, s))
    for group in s:
        idx = np.ix_(group,group)
        logp_H += compute_logp_independent_block_mc(N[idx], alpha[idx].sum(1), alpha[idx].sum(0), iterations=iterations)
    return logp_H


if __name__ == '__main__':

    np.random.seed(0)

    print __doc__
    
    # number of iterations of MC integration
    iterations = 1e5
    
    # confusion matrix ICANN 2012 - Team Huttunen
    N_hut = np.array([[94,  29, 16, 10,   1],
                      [22, 100, 10, 18,   1],
                      [25,  16, 51, 10,   0],
                      [ 3,   4, 12, 85,  21],
                      [ 2,   2,  4 , 3, 114]])
    
    # confusion matrix ICANN 2012 - Tu & Sun
    N_tu = tu = np.array([[56, 55, 36,  3,   0],
                          [30, 96, 21,  4,   0],
                          [33, 22, 46,  1,   0],
                          [ 4,  3,  3, 95,  20],
                          [ 1,  0,  0, 11, 113]])

    s_0 = [[0,1,2,3,4]]
    s_1 = [[0],[1],[2],[3],[4]]
    s_a = [[0,1,2],[3,4]]
    s_b = [[0,2],[1],[3],[4]]
    N_text = ['Huttunen','Tu & Sun']

    Ns = [N_hut,N_tu]
    s = [s_0,s_1,s_a,s_b]

    for i, N in enumerate(Ns):
        m = N.sum()
        alpha = np.ones(N.shape)
        c = N.shape[0]
        print "*", N_text[i]
        print "confusion matrix:"
        print N
        logp_H_mc_H_0 = compute_logp_H_mc(N, alpha, s[0], iterations=iterations)
        logp_H_mc_H_1 = compute_logp_H_mc(N, alpha, s[1], iterations=iterations)
        print
        
        # Binomial test
        binom_N = binom(m, 1.0/c)
        print 'p-value: %.3g' % binom_N.sf(np.trace(N))
        print
        
        # Bayes factor for independence vs. full dependence
        print "B_10:", np.exp(logp_H_mc_H_1 - logp_H_mc_H_0)
        print
        
        for j in range(2,len(s)):
            # log likelihood
            logp_H_mc = compute_logp_H_mc(N, alpha, s[j], iterations=iterations)
            print "s:", str(s[j]).replace('[','{').replace(']','}')
            print "log(p(N|s,alpha)): %s" % logp_H_mc
            # Bayes factor for partition
            print "B_1s:", np.exp(logp_H_mc_H_1 - logp_H_mc)
            print

        print "--------"
        print
