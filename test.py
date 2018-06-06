from sympy import *
import numpy as np
from tqdm import tqdm


class dh_solver():
    def __init__(self):
        self.joints_list = []
        self.T_list = []
        self.T = eye(4)
        self.T_sub = eye(4)

    def add(self, dh_param):
        if len(dh_param) != 4:
            raise ValueError('the number of input dh parameters !=4, it should be structure as [d,theta,a, alpha].')
        else:
            self.joints_list.append(dh_param)

    def dh_matrix(self, parameters):
        d, theta, a, alpha = parameters
        # first row
        dh11 = cos(theta)
        dh12 = -1 * sin(theta) * cos(alpha)
        dh13 = sin(theta) * sin(alpha)
        dh14 = a * cos(theta)
        # second row
        dh21 = sin(theta)
        dh22 = cos(theta) * cos(alpha)
        dh23 = -1 * cos(theta) * sin(alpha)
        dh24 = a * sin(alpha)
        # third row
        dh31 = 0
        dh32 = sin(alpha)
        dh33 = cos(alpha)
        dh34 = d
        # forth row
        dh41 = 0
        dh42 = 0
        dh43 = 0
        dh44 = 1
        return Matrix([[dh11, dh12, dh13, dh14],
                       [dh21, dh22, dh23, dh24],
                       [dh31, dh32, dh33, dh34],
                       [dh41, dh42, dh43, dh44]])

    def calc_symbolic_matrices(self):
        self.T_list = []
        self.T = eye(4)
        for i in tqdm(range(len(self.joints_list))):
            d = Symbol("d" + str(i + 1))
            theta = Symbol("theta" + str(i + 1))
            a = Symbol("a" + str(i + 1))
            alpha = Symbol("alpha" + str(i + 1))
            parameters = [d, theta, a, alpha]
            Tn = self.dh_matrix(parameters)
            self.T_list.append(Tn)
            self.T = self.T * Tn
        return self.T

    def calc_dh_matrix(self):
        self.calc_symbolic_matrices()
        self.T_sub = self.T
        for i in tqdm(range(len(self.joints_list))):
            ds = "d" + str(i + 1)
            thetas = "theta" + str(i + 1)
            a_s = "a" + str(i + 1)
            alphas = "alpha" + str(i + 1)
            parameters = [ds, thetas, a_s, alphas]
            for pair in zip(parameters, self.joints_list[i]):
                #                 print(pair)
                self.T_sub = self.T_sub.subs(pair[0], pair[1])
        return self.T_sub

    def get_numpy_matrix(self, list_subs):
        #         self.calc_dh_matrix()
        T = (self.T_sub.subs(list_subs)).evalf()
        return np.array(T.tolist()).astype(np.float64)

if __name__ == '__main__':
    from IPython.display import Latex
    import sympy
    from sympy import Symbol

    mtb = dh_solver()
    mtb.add([194, "theta1", 0, -sympy.pi / 2])
    # input a list [ d, theta, a, alpha]
    # or you can add the variable as a Sympy symbol, in this case you can also shift the variable
    mtb.add([0, "theta2", 149.5, 0])
    mtb.add([0, "theta3", 0, -sympy.pi / 2])
    mtb.add([204.5, "theta4", 0, sympy.pi / 2])
    mtb.add([0, "theta5", 0, -sympy.pi / 2])
    mtb.add([216, "theta6", 0, 0])
    T = mtb.calc_symbolic_matrices()
    print(T)
    # print(sympy.simplify(T))
    # print(mtb.T_list)
    T1 = mtb.calc_dh_matrix()
    T2 = sympy.simplify(T1)
    print(T2)
    a = sympy.latex(T2)
    print(Latex(a))
    arr = mtb.get_numpy_matrix([["theta1", sympy.pi*56 / 180], ["theta2", -sympy.pi / 2], ["theta3", sympy.pi*22 / 180], \
                                ["theta4", sympy.pi*45 / 180],["theta5", -sympy.pi*15 / 180],["theta6", sympy.pi*88 / 180]])
    print(arr)

