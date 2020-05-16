import pandas as pd 
import numpy as np
def get_predict(dp_model,batch):
    output = dp_model.predict_theta_from_input([batch[0]])
    sig = output[1][0,:,0]
    mu = output[0][0,:,0]
    tot_res = pd.DataFrame()
    tot_res['mu'] = mu
    tot_res['sigma'] = np.sqrt(sig)
    tot_res['upper'] = tot_res['mu']+tot_res['sigma']
    tot_res['lower'] = tot_res['mu']-tot_res['sigma']
    tot_res['two_upper'] = tot_res['mu']+2*tot_res['sigma']
    tot_res['two_lower'] = tot_res['mu']-2*tot_res['sigma']
    
    return tot_res