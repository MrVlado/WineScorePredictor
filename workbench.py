from svm import score
import util as u
import numpy as np

score_poly, model_poly = score("poly", "red")
score_rbf, model_rbf = score("rbf", "red")
score_sigmoid, model_sigmoid = score("sigmoid", "red")

print(model_rbf.get_params(), "   Score: ", score_rbf)
print(model_poly.get_params(), "   Score: ", score_poly)
print(model_sigmoid.get_params(), "   Score: ", score_sigmoid)
