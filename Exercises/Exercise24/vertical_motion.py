from numpy import *
import nose.tools as nt
from matplotlib.pyplot import *
class Problem:
    def __init__(self, m, T, V, d, C_D, A, g=9.81, rho = 10**3, mu = 8.9*10**-4,  v_e = None, v_0 = 0,t_0 = 0, StokesSource = None,QuadSource = None, ForceStep = None):
        self.g = g
        self.m = m
        self.T = T
        self.rho = rho
        self.V = V
        self.d = d
        self.mu = mu
        self.C_D = C_D
        self.A = A
        self.v_e = v_e
        self.v_0 = v_0
        self.t_0 = t_0

        if hasattr(rho,'__call__'):
            self.rho = rho
        else:
            self.rhoconst = rho
            self.rho = lambda t: self.rhoconst

        if StokesSource == None:
            self.ss = lambda t,a,b: 0
        else:
            self.ss = StokesSource

        if QuadSource == None:
            self.qs = lambda t,a,b: 0
        else:
            self.qs = QuadSource

        if ForceStep == 'Stokes':
            self.Reynold_number = lambda t,v: 0
        elif ForceStep == 'Quad':
            self.Reynold_number = lambda t,v: 2
        else:
            self.Reynold_number = self.Reynold_number_standard

    def exact_solution(self,t):
        return self.v_e(self,t)

    def stokes_coefficients(self,t):
        p = self
        rho_b = float(p.m)/p.V
        a = 3*pi*p.d*p.mu/(rho_b*p.V)
        btemp = p.g*(p.rho(t)/rho_b - 1)
        b = btemp + p.ss(t,a,btemp)
        return a,b

    def quad_coefficients(self,t):
        p = self
        rho_b = float(p.m)/p.V
        a = 0.5*p.C_D*p.rho(t)*p.A/(rho_b*p.V)
        btemp = p.g*(p.rho(t)/rho_b - 1)
        b = btemp + p.qs(t,a,btemp)
        return a,b
        
    def Reynold_number_standard(self,t,v):
        p = self
        if p.mu != 0:
            return p.rho(t)*p.d*abs(v)/float(p.mu)
        else:
            return 0

class Solver:
    def __init__(self,problem,dt):
        self.p = problem
        self.dt = dt

    def solve(self):
        p = self.p; dt = self.dt;
        n = int(round(p.T/dt))
        t = linspace(p.t_0, p.T, n+1)
        v = zeros(n+1)
        v[0] = p.v_0
        for i in range(0,n):
            if p.Reynold_number(t[i],v[i]) < 1:
                a, b = p.stokes_coefficients(t[i] + 0.5*dt)
                v[i+1] = ((1 - 0.5*a*dt)*v[i] + b*dt)/(1 + 0.5*a*dt)
            else:
                a, b = p.quad_coefficients(t[i] + 0.5*dt)
                v[i+1] = (v[i] + dt*b)/(1 + dt*a*abs(v[i]))
        self.v = v; self.t = t; self.n = n

class Tester:

    #Tests if exact solution is reporduced to machine precision when drag force is 0
    def test_heavy_sphere(self):
        r = 2;d = 2*r;m = 2.0;T = 3; V = 4/3*pi*r**3; A = pi*r**2;C_D = 0.45;
        def v_e(problem,t):
            p = problem; g = p.g;v_0 = p.v_0;
            return v_0 - g*t
        sphere = Problem(m, T, V, d, C_D, A, mu=0, rho = 0,v_e = v_e)
        s = Solver(sphere,0.1)
        s.solve()
        maxError = max(s.v - s.p.exact_solution(s.t))
        assert(maxError <= 10**-10) 

    #Tests the terminal velocity for mostly large Reynold's numbers
    def test_terminal_velocity_quad(self):
        r = 0.05;d = 2*r;T = 10; V = 4/3*pi*r**3;m = (10**4)*V; A = pi*r**2;C_D = 0.45;rho = 10**3;mu = 8.9*10**-4
        sphere = Problem(m, T, V, d, C_D, A,rho=rho,mu=mu)
        s = Solver(sphere,0.1)
        s.solve()
        a,b = sphere.quad_coefficients(T)
        v_e =  sign(b)*sqrt(abs(b)/a)
        assert(abs(s.v[-1] - v_e) < 10**-10)

    #Tests the terminal velociy for small Reynold's numbers
    def test_terminal_velocity_stokes(self):
        r = 0.05;d = 2*r;T = 0.01; V = 4/3*pi*r**3;m = (10**3)*V; A = pi*r**2;C_D = 0.45;rho = 1.2;mu = 1.9*10**3
        sphere = Problem(m, T, V, d, C_D, A,rho=rho,mu=mu,v_0 = 5)
        dt = 1.0/10000
        s = Solver(sphere,dt)
        s.solve()
        a,b = sphere.stokes_coefficients(T)
        #v_e =  sign(b)*sqrt(abs(b)/a)
        v_e = b/a
        assert(abs(s.v[-1] - v_e) < 10**-10)

        
#Tests if linear solutions is reproduced to machine precision
    def test_linear(self):
        for step in ['Stokes','Quad','Mixed']:
            r = 0.05;d = 2*r;T = 1; V = 4/3*pi*r**3;m = (10**4)*V; A = pi*r**2;C_D = 0.45;rho = 10**3;mu = 8.9*10**-4; dt = 0.1;

            c1 = 1; c2 = 0;
            u = lambda t: c1*t + c2
            bs = lambda t,a,b: -b + c1 + a*u(t)
            bq = lambda t,a,b: -b + c1 + a*u(t)*abs(u(t)) - 0.25*a*dt**2
            sphere = Problem(m, T, V, d, C_D, A,rho=rho,mu=mu, v_0 = c2,StokesSource = bs,QuadSource = bq,ForceStep = step)
            s = Solver(sphere,dt)
            s.solve()
            a,b = sphere.quad_coefficients(T)
            v_e =  u(T)
            assert(abs(s.v[-1] - v_e) < 10**-13)

#Convergence test on quadratic solutions
    def test_quadratic(self):
        e = []; dta = [0.01, 0.001]
        for i in range(0,len(dta)):
            r = 0.05;d = 2*r;T = 1; V = 4/3*pi*r**3;m = (10**4)*V; A = pi*r**2;C_D = 0.45;rho = 10**3;mu = 8.9*10**-4;dt = dta[i]

            c1 = 1; c2 = 2;c3 = 0;
            u = lambda t: c1*t**2 + c2*t + c3
            bs = lambda t,a,b: -b + 2*c1*t + c2 + a*u(t)
            bq = lambda t,a,b: -b + 2*c1*t + c2 + a*u(t)*abs(u(t))
            sphere = Problem(m, T, V, d, C_D, A,rho=rho,mu=mu, v_0 = c3,StokesSource = bs,QuadSource = bq)
            s = Solver(sphere,dt)
            s.solve()
            a,b = sphere.quad_coefficients(T)
            v_e =  u(T)
            e.append(s.v[-1] -  v_e)
        r = log(e[0]/e[1])/log(dta[0]/dta[1])
        assert(r - 2 < 10**-5)

if __name__ == '__main__':
        r = 0.11;d = 2*r;T = 0.1; V = 4/3*pi*r**3;m = 0.43; A = pi*r**2;C_D = 0.45;rho = 10**3;mu = 8.9*10**-4
        sphere = Problem(m, T, V, d, C_D, A,rho=rho,mu=mu)
        s = Solver(sphere,0.001)
        s.solve()
        plot(s.t,s.v)
        xlabel('Time (s)')
        ylabel('Velocity (m/s)')
        title('Ball in water')
        show()
        
