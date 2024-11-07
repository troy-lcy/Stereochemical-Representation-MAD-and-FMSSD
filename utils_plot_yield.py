import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import matplotlib.colors as mcolors

# Import relevant scikit-learn modules
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from torch.autograd import Variable
import torch

def fit_models(X_train, 
               X_test,
               y_train, 
               y_test,
               models=[]):
    predictions = []
    r2_values = []
    rmse_values = []
    for model in models:
        print(model)
        # fit the model and generate predictions
        model.fit(X_train, y_train.ravel())
        preds = model.predict(X_test)

        # calculate an R-squared and RMSE values
        r_squared = r2_score(y_test, preds)
        rmse = mean_squared_error(y_test, preds) ** 0.5

        # append all to lists
        predictions.append(preds)
        r2_values.append(r_squared)
        rmse_values.append(rmse)
    print('Done fitting models')
    return predictions, r2_values, rmse_values

# 有的模型是提前训练好的直接加载来测试，有的是没训练过现在要训练
def load_or_fit_models(X_train, 
               X_test,
               y_train, 
               y_test,
               models_need_fit=[], models_already_fitted=[]):
    predictions = []
    r2_values = []
    rmse_values = []
#     models = []#所有模型，包括训练过和现在要训练的
#     for model in models_need_fit:
#         print(model)
#         # fit the model and generate predictions
#         model.fit(X_train, y_train.ravel())
#         models.append(model)
#     
#     models.extend(models_already_fitted)
        
    # models_need_fit中的都是sklearn创建的模型
    #if False:
    for model in models_need_fit:
        print(model)
        # fit the model and generate predictions
        model.fit(X_train, y_train.ravel())
        preds = model.predict(X_test)

        # calculate an R-squared and RMSE values
        r_squared = r2_score(y_test, preds)
        rmse = mean_squared_error(y_test, preds) ** 0.5

        # append all to lists
        predictions.append(preds)
        r2_values.append(r_squared)
        rmse_values.append(rmse)

    # models_already_fitted中的暂时都是用pytorch创建的模型
    for model in models_already_fitted:
        print(model)
        # fit the model and generate predictions
        #model.fit(X_train, y_train.ravel())
        features_test = Variable(torch.from_numpy(X_test).to(torch.float))                    
        # Forward propagation
        outputs_test = model(features_test)
        preds = outputs_test.data
        #preds = model.predict(X_test)

        # calculate an R-squared and RMSE values
        r_squared = r2_score(y_test, preds)
        rmse = mean_squared_error(y_test, preds) ** 0.5

        # append all to lists
        predictions.append(preds)
        r2_values.append(r_squared)
        rmse_values.append(rmse)
    print('Done fitting models')
    return predictions, r2_values, rmse_values


def plot_models(predictions,
                r2_values,
                rmse_values,
                y_test,
                titles =[ 'AdaBoost',
                          'Linear Regression',
                          'Support Vector Machine',
                          'k-Nearest Neighbors',
                          'Random Forest',
                          'Neural Network'],
                positions=[231,232,233,234,235,236],
                colors = [1,2,3,4,5,6],
                save=False):

    fig = plt.figure(figsize=(15,10))
    for pos, pred, r2, rmse, title ,color in zip(positions,
                                          predictions,
                                          r2_values,
                                          rmse_values,
                                          titles,
                                          colors):
        # create subplot
        plt.subplot(pos)
        plt.grid(alpha=0.2)
        plt.title(title, fontsize=15)
        colors=list(mcolors.TABLEAU_COLORS.keys())
        # add score patches
        r2_patch = mpatches.Patch(label="R2 = {:04.2f}".format(r2),color=mcolors.TABLEAU_COLORS[colors[color]])
        rmse_patch = mpatches.Patch(label="RMSE = {:04.1f}".format(rmse),color=mcolors.TABLEAU_COLORS[colors[color]])
        # plt.xlim(-40,130)
        # plt.ylim(-10,130)
        plt.scatter(pred, y_test, alpha=0.2,color=mcolors.TABLEAU_COLORS[colors[color]])
        plt.legend(handles=[r2_patch, rmse_patch], fontsize=12,loc='upper left')
        plt.plot(np.arange(100), np.arange(100), ls="--", c=".3")
        fig.text(0.5, 0.07, 'predicted yield%', ha='center', va='center', fontsize=15)
        fig.text(0.09, 0.5, 'observed yield%', ha='center', va='center', rotation='vertical', fontsize=15)
        plt.savefig('compare.png', dpi = 300)
    plt.show()