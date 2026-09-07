import torch as to
import numpy as np
class Linear:
    def __init__(self,entradas=1, saidas=1):
        self.neoronio=to.nn.Linear(entradas, saidas)
        self.criterio=to.nn.MSELoss()
        self.optimi=to.optim.SGD(self.neoronio.parameters(), 0.01)
    def fit(self, X, Y):
        X = to.as_tensor(X, dtype=to.float32)
        Y = to.as_tensor(Y, dtype=to.float32)
        for i in range(1000):
            prev=self.neoronio(X)
            erro=self.criterio(prev, Y)
            erro.backward()
            self.optimi.step()
            self.optimi.zero_grad()
            if erro.item() <= 0.001:
                break
    def predict(self, X):
        X = to.as_tensor(X, dtype=to.float32)

        return self.neoronio(X)
            

        


        

