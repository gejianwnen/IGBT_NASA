from matplotlib import pyplot as plt

def plot_batch(batch, TIME_STEPS):
    plt.plot(batch[1].reshape(TIME_STEPS), linewidth=3)
    
def plot_predict(tot_res):
#     plt.plot(tot_res.mu, 'bo')
    plt.plot(tot_res.mu, linewidth=2)
    plt.legend("real","predeict")
    plt.fill_between(x = tot_res.index, y1=tot_res.lower, y2=tot_res.upper, alpha=0.5)
    plt.fill_between(x = tot_res.index, y1=tot_res.two_lower, y2=tot_res.two_upper, alpha=0.5)
    plt.title('Prediction uncertainty')
    return 0