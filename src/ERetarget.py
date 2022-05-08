import numpy as np

from src.mesh import triangulate_vertices
from src.mesh import build_Laplacian


class ERetarget():
    """
    Construct a class to compute E_Retarget as in formula 16 and applay facial retargetting on each frame of a sequence
    by minimizing the weights vector w

    k:= num_of_blendshapes
    f:= num_frames
    m:= num_markers
    n:= num_features

    """
    def __init__(self, dp, LdV, mu=0.3, nu=0.6):
        self.mu = mu
        self.nu = nu
        self.K = np.shape(dp)[0]
        self.n = np.shape(dp)[1]
        self.M = int(self.n / 3)
        self.N = np.shape(LdV)[1]
        self.V = int(self.N / 3)

        self.af = None
        self.delta_p = dp
        self.LdV = LdV

    def set_af(self, af):
        """
        Enable to set the actor frame

        :param af: actor frame
        :return:
        """
        self.af = af

    def _e_fit(self, w):
        """
        Compute EFit as in formula 2

        :param w: vector of size (k,)
        :return:
        """

        w = np.repeat(np.expand_dims(w, axis=1), self.n, axis=1)
        w_comb = np.multiply(w, self.delta_p)
        fits = self.af - np.sum(w_comb, axis=0)

        return np.linalg.norm(fits)**2/self.M

    def _e_sparse(self, w):
        """
        Compute ESparse as (1/K) * ||w||1  (1-norm)
        :param w:
        :return:
        """
        return np.linalg.norm(w, ord=1) / self.K

    def _build_L_deltaV(self, v, v0):
        """
        Compute the matrix product L * delta_V as it remains constant over time
        delta_V = v - v0

        :param v: expression of interest
        :param v0: neutral expression
        :return:
        """
        LdV = []
        for k in range(self.K):
            # compute delta_V
            if np.array_equal(v[k], v0):
                # avoid the case of v[k] = v0
                dV = v[k]
            else:
                dV = v[k] - v0
            # build mesh
            mesh = triangulate_vertices(dV)
            # build Laplacian
            L = build_Laplacian(mesh, n_v)
            L = L.todense()
            L_expand = np.zeros((n_n, n_n))
            # expand L by 3
            for i in range(n_n):
                for j in range(n_n):
                    if i % 3 == 0 and j % 3 == 0:
                        L_expand[i, j] = L[int(i / 3), int(j / 3)]
                    if i % 3 == 1 and j % 3 == 1:
                        L_expand[i, j] = L[int(i / 3), int(j / 3)]
                    if i % 3 == 2 and j % 3 == 2:
                        L_expand[i, j] = L[int(i / 3), int(j / 3)]
            LdV.append(L_expand @ dV)

        return np.array(LdV)

    def _e_prior(self, w):
        """
        compute EPrior as in formula 15

        EPrior(w) = (1/N) * ||L*delta_V*w||^2

        :param w: vector weights of size (k,)
        :return:
        """
        prior = np.multiply(self.LdV, np.repeat(np.expand_dims(w, axis=1), self.N, axis=1))
        return np.sum(np.linalg.norm(prior)**2) / self.N

    def _e_retarget(self, w):
        """
        Compute eRetarget as in formula 16

        ERetarget(w) = EFit(w) + mu*EPrior(w) + nu*ESparse(w)

        :param w: vector weights of size (k,)
        :return:
        """
        return self._e_fit(w) + self.mu * self._e_prior(w) + self.nu * self._e_sparse(w)

    def get_EFit(self):
        """
        return EFit as a function

        :return:
        """
        return self._e_fit

    def get_EPrior(self):
        """
        return EPrior as a function

        :return:
        """
        return self._e_prior

    def get_eRetarget(self):
        """
        return ERetarget as a function

        :return:
        """
        return self._e_retarget

    def get_dEFit(self):
        """
        Compute the derivative of E_fit (formula 2) and split the equation to fit the form: Ax + b
        With A a square matrix of size (kxk) and b a vector of size (k,)

        k:= num blendshapes
        M := num_markers

        :return: A, b
        """
        A = (2/self.M) * self.delta_p @ self.delta_p.T
        b = (2/self.M) * self.delta_p @ self.af

        return A, b

    def get_dEPrior(self):
        """
        Compute the equation system to solve EPrior (formula 15)

        EPrior(w) = (1/N) * ||L*delta_V*w||^2

        :return: A, b
        """
        A = (2/self.N) * np.diag(np.sum(np.power(self.LdV, 2), axis=1))
        b = np.zeros(self.K)

        return A, b

    def get_dESparse(self):
        """
        Compute the equation system to solve ESparse

        :return: A, b
        """
        A = (1/self.K) * np.eye(self.K)
        b = np.zeros(self.K)

        return A, b

    def get_dERetarget(self):
        """
        Return the equation system to solve ERetarget as formula 16.
        It adds up EFit, EPrior and ESparse in a square matrix A and a vector b as to solve the equation Ax + b

        :return:
        """
        AFit, bFit = self.get_dEFit()
        APrior, bPrior = self.get_dEPrior()
        ASparse, bSparse = self.get_dESparse()

        A = AFit + self.mu * APrior + self.nu * ASparse
        b = bFit + self.mu * bPrior + self.nu * bSparse

        return A, b


if __name__ == '__main__':
    """
    test ERetarget
    
    1) compute and test EFit energy error
    2) compute and test EFit function (minimize) vs. EFit equation (solve)
    3) compute and test EPrior energy error
    4) compute and test EPrior function (minimize) vs. EPrior equation (solve)
    5) compute and test ERetarget function (minimize) vs. ERetarget equation (solve)
    
    run: python -m src.ERetarget
    """
    import time as time
    from scipy import optimize
    from scipy.linalg import solve

    np.random.seed(1)
    np.set_printoptions(precision=4, linewidth=200, suppress=True)
    # declare variables
    n_k = 4  # num_blendshapes
    n_f = 1  # num_frames
    n_m = 5  # num_markers (min 4 to use Delaunay)
    n_n = n_m * 3  # num_features (sparse)
    n_v = 40  # num_vertices (min 4 to use Delaunay)
    n_N = n_v * 3  # num_features (full)

    af = np.random.rand(n_n)  # one single frame!
    dpk = np.random.rand(n_k, n_n)
    w = np.random.rand(n_k)  # only a single weights per blendshapes!
    LdV = np.random.rand(n_k, n_N)
    print("shape af", np.shape(af))
    print("shape dpk", np.shape(dpk))
    print("shape w", np.shape(w))
    print("shape LdV", np.shape(LdV))
    print()

    # declare e_retarget
    e_retarg = ERetarget(dpk, LdV)
    e_retarg.set_af(af)

    # ------------------------------------------------------------------------------
    # ---------------------------      E Fit         -------------------------------
    # ------------------------------------------------------------------------------
    print("-------- EFit ---------")
    # test eFit
    fits = []
    for k in range(n_k):
        fit = w[k] * dpk[k]
        fits.append(fit)
    # np.sum(fits, axis = 0) : (15,1)
    # => sk 별로 marker 별로 xyz에 각각 같은 weight w[k]가 곱해진 값을 row를 축으로 sum해낸 값, 즉 (1x15) row vector가 출력된다.
    print(np.sum(fits,axis=0))

    # 위의 값을 frame별 sparse actor representation에서 빼준 값을 fits로 재정의
    fits = af - np.sum(fits, axis=0)
    e_fit_test = np.linalg.norm(fits)**2/n_m
    print("[EFit]e_fit_test", e_fit_test)

    e_fit = e_retarg._e_fit(w)
    print("[EFit]e_fit", e_fit)
    # assert는 올바른 상황을 명시, 그 상황에 어긋날 경우, error 도출
    assert e_fit == e_fit_test
    print("[EFit] Error values are equal")
    print()

    print("[EFit]----- Minimization ------")
    print("[EFit]try optimizer")
    start = time.time()
    opt = optimize.minimize(e_retarg.get_EFit(), w, method="BFGS")
    print("[EFit]solved in:", time.time() - start)
    print(opt.x) # dpk, sk 모두 4개씩이므로 4차원 vector가 출력된다

    print("[EFit]try solver")
    A, b = e_retarg.get_dEFit()
    start = time.time()
    sol = solve(A, b)
    print("[EFit]solved in:", time.time() - start)
    print("[EFit]Sol")
    print(sol)

    # test if values matches
    np.testing.assert_array_equal(np.around(opt.x, 4), np.round(sol, 4))
    print("[EFit] Optimization vs. Solver reaches same values!")
    print()

    # this is where this paper differs from other approaches
    # ------------------------------------------------------------------------------
    # ---------------------------      E Prior       -------------------------------
    # ------------------------------------------------------------------------------
    print("-------- EPrior ---------")

    # test ePrior
    priors = []
    for k in range(n_k):
        # compute prior energy
        prior = LdV[k] * w[k]
        prior = np.linalg.norm(prior)**2
        priors.append(prior)
    ePrior_test = np.sum(priors) / n_N
    print("[EPrior] EPrior_test", ePrior_test)

    # compute e_prior
    e_prior_fn = e_retarg.get_EPrior()
    ePrior = e_prior_fn(w)
    print("[EPrior] ePrior")
    print(ePrior)

    assert round(ePrior_test, 5) == round(ePrior, 5)
    print("[EPrior] Error values are equal")
    print()

    print("[EPrior] ----- Minimization ------")
    print("[EPrior] try optimizer")
    start = time.time()
    opt = optimize.minimize(e_retarg.get_EPrior(), w, method="BFGS")
    print("[EPrior] solved in:", time.time() - start)
    print(opt.x)

    print("[EPrior] try solver")
    A, b = e_retarg.get_dEPrior()
    start = time.time()
    sol = solve(A, b)
    print("[EPrior] solved in:", time.time() - start)
    print("[EPrior] shape sol", np.shape(sol))
    print(sol)

    # test if values matches
    np.testing.assert_array_equal(np.around(opt.x, 5), np.round(sol, 5))
    print("Reached same value!")

    # ------------------------------------------------------------------------------
    # ---------------------------      E Retarget    -------------------------------
    # ------------------------------------------------------------------------------
    # E Retarget = E Fit + E Reg
    print("-------- ERetarget ---------")

    print("[ERetarget] try solver")
    A, b = e_retarg.get_dERetarget()
    start = time.time()
    sol = solve(A, b)
    print("[ERetarget] solved in:", time.time() - start)
    print("[ERetarget] shape sol", np.shape(sol))
    print(sol)

    print("[ERetarget] test minimize")
    start = time.time()
    # Minimization of scalar function of one or more variables using the BFGS algorithm.
    # with initial guess sol
    opt = optimize.minimize(e_retarg.get_eRetarget(), sol, method="BFGS")
    print("[ERetarget] solved in:", time.time() - start)
    print("[ERetarget] shape opt.x", np.shape(opt.x))
    print(opt.x)

    print("[ERetarget] Least Square")
    start = time.time()
    #  least_squares finds a local minimum of the cost function F(x)
    # sol : 초깃값, Initial guess on independent variables. If float, it will be treated as a 1-D array with one element.
    # BFGS 알고리즘 : f(x)를 제한 조건이 없는 실수 벡터 x에 대해서 최소화 시키는 것으로, 여기서 함수 f(x)는 스칼라 함수가 되겠다.
    lsq = optimize.least_squares(e_retarg.get_eRetarget(), sol)
    print("[ERetarget] solved in:", time.time() - start)
    print("[ERetarget] shape lsq", np.shape(lsq.x))
    print(lsq.x)

    print("eRetarget optimized:", e_retarg._e_retarget(opt.x))
    print("eRetarget solved:", e_retarg._e_retarget(sol))
    print("eRetarget least square:", e_retarg._e_retarget(lsq.x))

