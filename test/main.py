from pycaret.datasets import get_data
from pycaret.regression import *
data = get_data('insurance')
setup(data, target='charges', session_id=123)

best = compare_models()
print(best)

plot_model(best, plot='residuals', save=True)
plot_model(best, plot='error')
plot_model(best, plot='cooks')
plot_model(best, plot='rfe')
plot_model(best, plot='learning')
plot_model(best, plot='vc')
plot_model(best, plot='feature')
