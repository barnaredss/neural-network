from typing import Callable
import jax
import numpy as np

class UOSolver:
    def __init__(
        self,
        f: Callable,
        direction: str,
        dimensions: int,
        g: Callable = None,
        h: Callable = None,
        max_iter: int = 2000,
        lr: float = 0.001,
        epsilon: float = 0.001,
        x0: np.ndarray = None,
        momentum_coef: float = 0.75,
        bls_rho: float = 0.5,
        alpha_max: float = 1.0,
        alpha_min: float = 0.01,
        wolf_c1: float = 0.1,
        wolf_c2: float = 0.9,
        H: np.ndarray = None,
        adam_alpha: float = 0.001,
        beta1_adam: float = 0.9,
        beta2_adam: float = 0.999,
        weight_decay: float = 0.01,
        ) -> None:

        self.dim = dimensions
        self.direction = direction
        self.d = np.zeros(dimensions)
        self.alpha = lr
        self.f = f
        self.g = g if g is not None else jax.grad(f)
        self.current_g = np.zeros(dimensions)
        self.h = h if h is not None else jax.hessian(f)
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.k = 1
        self.x = x0 if x0 is not None else np.zeros(dimensions)
        self.x_1 = self.x.copy()
        self.momentum_coef = momentum_coef
        self.bls_rho = bls_rho
        self.alpha_max = alpha_max
        self.alpha_min = alpha_min
        self.wolf_c1 = wolf_c1
        self.wolf_c2 = wolf_c2
        if self.direction == "bfgs":
            self.H = H if H is not None else np.eye(dimensions)
            self.bfgs_s = np.zeros(dimensions)
            self.bfgs_y = np.zeros(dimensions)
            self.prev_g = np.zeros(dimensions) 
        else:
            self.H = None 
            self.bfgs_s = None
            self.bfgs_y = None
            self.prev_g = None
        self.bfgs_rho = 0.0
        self.adam_alpha = adam_alpha
        self.beta1_adam = beta1_adam
        self.beta2_adam = beta2_adam
        self.m_adam = np.zeros(dimensions)
        self.v_adam = np.zeros(dimensions)
        self.adam_eps = 1e-8
        self.weight_decay = weight_decay

    def uosol(self) -> np.ndarray:
        """Solves unconstrained optimization returning a minimizer"""

        while self.k < self.max_iter and np.linalg.norm(self.g(self.x)) > self.epsilon:
            self.d, self.alpha = self.step(self.g(self.x))
            self.x_1 = self.x.copy()
            self.x -= self.alpha * self.d
            self.k += 1

        return self.x

    def get_direction_adam(self, gradient: np.ndarray) -> np.ndarray:
        """Gets descent direction for adam method"""

        self.m_adam = self.beta1_adam * self.m_adam + (1 - self.beta1_adam) * gradient          # m will track direction of descent
        self.v_adam = self.beta2_adam * self.v_adam + (1 - self.beta2_adam) * gradient ** 2     # v will track magnitude of descent
        m_hat = self.m_adam / (1 - self.beta1_adam ** self.k)
        v_hat = self.v_adam / (1 - self.beta2_adam ** self.k)
        return m_hat / (np.sqrt(v_hat) + self.adam_eps)

    def get_direction_adamw(self, gradient: np.ndarray) -> np.ndarray:
        """Gets descent direction for AdamW (Adam with decoupled weight decay)"""

        self.m_adam = self.beta1_adam * self.m_adam + (1 - self.beta1_adam) * gradient
        self.v_adam = self.beta2_adam * self.v_adam + (1 - self.beta2_adam) * gradient ** 2
        m_hat = self.m_adam / (1 - self.beta1_adam ** self.k)
        v_hat = self.v_adam / (1 - self.beta2_adam ** self.k)
        return m_hat / (np.sqrt(v_hat) + self.adam_eps) + self.weight_decay * self.x


    def get_direction_sgd_momentum(self, gradient: np.ndarray) -> np.ndarray:
        """Gets descent direction for sgd momentum method"""

        return self.momentum_coef * self.d + gradient

    def get_direction_bfgs(self, gradient: np.ndarray) -> np.ndarray:
        """Gets descent direction for BFGS method using inverse Hessian approximation"""

        d = self.H @ gradient
        self.bfgs_s = self.x - self.x_1
        self.bfgs_y = gradient - self.prev_g
        rho_denom = np.dot(self.bfgs_y, self.bfgs_s)
        if abs(rho_denom) > 1e-10:
            self.bfgs_rho = 1.0 / rho_denom
            A = np.eye(self.dim) - self.bfgs_rho * np.outer(self.bfgs_s, self.bfgs_y)
            B = np.eye(self.dim) - self.bfgs_rho * np.outer(self.bfgs_y, self.bfgs_s)
            self.H = A @ self.H @ B + self.bfgs_rho * np.outer(self.bfgs_s, self.bfgs_s)
        return d

    def get_direction_newton_cg(self, gradient: np.ndarray) -> np.ndarray:
        """Gets descent direction via Newton CG method solving H @ d = g"""

        H = self.h(self.x)
        g_norm = np.linalg.norm(gradient)
        tolerance = min(0.5, np.sqrt(g_norm)) * g_norm  # Tighter tolerance near the solution
        d = np.zeros(self.dim)
        r = gradient.copy()   # residual (g - H @ 0 = g)
        p = r.copy()          # CG search direction
        r_dot = np.dot(r, r)

        for i in range(self.dim):
            Hp = H @ p
            pHp = np.dot(p, Hp)
            if pHp <= 0:    # Non-positive curvature
                return gradient if i == 0 else d

            alpha = r_dot / pHp
            d = d + alpha * p
            r = r - alpha * Hp

            r_dot_new = np.dot(r, r)
            if np.sqrt(r_dot_new) <= tolerance:
                break
            beta = r_dot_new / r_dot
            p = r + beta * p
            r_dot = r_dot_new
        return d


    def _is_in_ac(self) -> bool:
        """Returns whether self.alpha is in Acceptability Conditions"""
        
        x_new = self.x - self.alpha * self.d
        current_grad_dot_d = np.dot(self.current_g, self.d)
        wc1 = self.f(x_new) <= self.current_f - self.wolf_c1 * self.alpha * current_grad_dot_d
        wc2 = abs(np.dot(self.g(x_new), self.d)) <= self.wolf_c2 * abs(current_grad_dot_d)
        
        return wc1 and wc2
    
    def _backtracking_line_search(self) -> float:
        """Performs BLS to find optimal or good alpha"""

        self.alpha = self.alpha_max
        self.current_f = self.f(self.x)
        while self.alpha > self.alpha_min and not self._is_in_ac():
            self.alpha *= self.bls_rho
        return self.alpha

    def step(self, gradient: np.ndarray) -> tuple[np.ndarray, float]:
        """Returns direction vector and step length for a given gradient"""

        self.current_g = gradient

        if self.direction == "adam":
            self.d = self.get_direction_adam(gradient)
            self.alpha = self.adam_alpha

        elif self.direction == "adamw":
            self.d = self.get_direction_adamw(gradient)
            self.alpha = self.adam_alpha

        elif self.direction == "sgdm":
            self.d = self.get_direction_sgd_momentum(gradient)

        elif self.direction == "bfgs":
            self.d = self.get_direction_bfgs(gradient)
            self.alpha = self._backtracking_line_search()
            
        elif self.direction == "newtoncg":
            self.d = self.get_direction_newton_cg(gradient)
            self.alpha = self._backtracking_line_search()

        self.prev_g = self.current_g.copy()
        return self.d, self.alpha


if __name__ == "__main__":

    def f(x):
        return x[0]**2 + 5*x[1]**2

    def g(x):
        return np.array([2*x[0], 10*x[1]])

    solver = UOSolver(f=f, g=g, direction="sgdm", dimensions=2, x0=np.array([3.0, 3.0]))
    print(solver.uosol())
