import numpy as np
from tqdm import tqdm
from scipy.special import expit
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


class modelWrapper():
    def __init__(self, x, y, model=None):
        self.model = model or KNeighborsClassifier(n_neighbors=1)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=0.33)
        self.init_coef_mat()

    def evaluate(self, features):
        self.model.fit(self.x_train[:, np.where(features)[0]], self.y_train)
        return np.sum(self.model.predict(self.x_test[:, np.where(features)[0]]) == self.y_test) / len(self.x_test)

    def init_coef_mat(self):
        _, n = self.x_train.shape
        c = np.zeros(shape=(n, n))
        # TODO this can be vectorized
        for i in range(n):
            for j in range(n):
                x, y = self.x_train[:, i], self.x_train[:, j]
                c[i][j] = np.sum((x - x.mean()) *
                                 (y - y.mean()) / np.std(x) / np.std(y))
        self.coef_mat = c
        self.corr = np.sum(c, axis=1) / (n - 1)


class Particle:

    def __init__(self, dimension):
        self.position = np.random.randint(0, 2, size=(dimension))
        self.velocity = np.zeros(dimension)
        self.memory = self.position


class PSO:

    def __init__(self,
                 num_particles=50,
                 dimension=30,
                 model=None):

        self.model = model
        self.dimension = dimension
        self.particles = [Particle(dimension) for _ in range(num_particles)]
        self.update_global()

    def optimize(self, num_iter=100, w=0.8, c1=2, c2=2):

        for _ in tqdm(range(num_iter)):
            self.update_particles(w, c1, c2)
        return self.best

    def update_particles(self, w, c1, c2):
        for p in self.particles:
            self.update_position(p, w, c1, c2)
            self.update_memory(p)
        self.update_global()

    def update_position(self, p, w, c1, c2):
        r1, r2 = np.random.rand(2)
        p.velocity = w * p.velocity + r1 * c1 * (p.memory - p.position) + \
            r2 * c2 * (self.best.position - p.position)
        p.velocity = np.clip(p.velocity, -4, 4)
        p.position = (np.random.rand(self.dimension)
                      < expit(p.velocity)).astype(int)

    def update_memory(self, p):
        if self.model.evaluate(p.position) > self.model.evaluate(p.memory):
            p.memory = p.position

    def update_global(self):
        best = self.particles[0]
        max_score = self.model.evaluate(best.position)
        for p in self.particles:
            score = self.model.evaluate(p.position)
            if score > max_score:
                max_score = score
                best = p
        self.best = best


class HPSOLS(PSO):

    def __init__(self,
                 num_particles=50,
                 dimension=30,
                 model=None,
                 eps=None):

        super(HPSOLS, self).__init__(num_particles, dimension, model)
        self.init_num_features(eps)
        self.init_feature_set()

    def optimize(self, num_iter=10, alpha=0.65, w=0.8, c1=2, c2=2):
        for _ in tqdm(range(num_iter)):
            self.update_particles(w, c1, c2)
            self.move_particles(alpha)
        print(self.model.evaluate(self.best.position))
        return self.best

    def move_particles(self, alpha):
        for p in self.particles:
            self.move_particle(p, alpha)

    def move_particle(self, p, alpha) :

        nd, ns = int(np.ceil(alpha * self.num_features)),\
                 int(np.floor((1 - alpha) * self.num_features))

        f = np.where(p.position) 
        fd = list(np.intersect1d(self.dissimilars, f))
        fs = list(np.intersect1d(self.similars, f))

        fd = self.add_features(nd, fd, self.dissimilars)
        fd = self.delete_features(nd, fd, self.dissimilars)

        fs = self.add_features(ns, fs, self.similars[::-1])
        fs = self.delete_features(ns, fs, self.similars[::-1])

        p.position = np.zeros(self.dimension)
        p.position[fs + fd] = 1 

    def add_features(self, n, f, s):
        i = len(s) - 1
        while len(f) < n : 
            while s[i] in f : 
                i -= 1 
            f.append(s[i])
        return f 

    def delete_features(self, n, f, s):
        i = 0
        while len(f) > n : 
            while s[i] not in f : 
                i += 1 
            f.remove(s[i]) 
        return f 

    def init_feature_set(self):
        c = self.model.corr
        c = np.argsort(c)
        self.similars = c[len(c) // 2:]
        self.dissimilars = c[:len(c) // 2][::-1]

    def init_num_features(self, eps):

        eps = self.init_eps(eps)
        f = self.dimension
        sf = np.arange(3, f)
        denom = abs(np.cumprod(sf[::-1]))
        prob = (f - sf) / denom
        k = int(eps * f)
        prob = prob[:k]
        prob /= np.sum(prob)
        self.num_features = np.random.choice(sf[:k], p=prob)

    def init_eps(self, eps):
        if not eps:
            eps = 0
            while eps < 0.15 or eps > 0.7:
                eps = np.random.rand(1)
        return eps


data = load_breast_cancer()
x, y = data.data, data.target
model = modelWrapper(x, y)
p = HPSOLS(30, 30, model)
p.optimize()
