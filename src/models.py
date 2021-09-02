from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import NeuralNet

models = {'XGB': XGBRegressor(objective='reg:squarederror'), 'RF': RandomForestRegressor(), 'NN': NeuralNet.model}
