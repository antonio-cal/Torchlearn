import torch as to
import numpy as np
class Linear:
    def __init__(self,entradas=1, saidas=1):
        self.neoronio=to.nn.Linear(entradas, saidas)
        self.criterio=to.nn.MSELoss()
        self.optimi=to.optim.SGD(self.neoronio.parameters(), 0.01)
    def fit(self, X, Y):
        for i in range(1000):
            if isinstance(X, np.ndarray):
                X = to.from_numpy(X).float()
            else:
                X = X.float()

            if isinstance(Y, np.ndarray):
                Y = to.from_numpy(Y).float()
            else:
                Y = Y.float()
            if isinstance(X, list):
                X= to.tensor(X).float()
            if isinstance(Y, list):
                Y=to.tensor(Y).float()
            prev=self.neoronio(X)
            erro=self.criterio(prev, Y)
            erro.backward()
            self.optimi.step()
            self.optimi.zero_grad()
    def predict(self, X):
        if isinstance(X, np.ndarray):
            X = to.from_numpy(X).float()

        elif isinstance(X, (int, float)):
            X = to.tensor([[X]], dtype=to.float32)

        elif isinstance(X, to.Tensor):
            X = X.float()

        return self.neoronio(X)
            

        


        

