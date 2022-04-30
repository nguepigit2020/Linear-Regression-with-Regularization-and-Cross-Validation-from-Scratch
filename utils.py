import matplotlib.pyplot as plt
import numpy as np

def plot_loss_and_params(model, train_losses, test_losses):
    n = np.random.randint(1,20)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,4))
    ax1.plot(train_losses[:], label="train-loss", color='blue')
    ax1.plot(test_losses[:], label="test-loss", color='red')
    ax1.set_title('loss')
    ax2.stem(model.theta, use_line_collection=True)
    ax2.set_title('params values')
    ax3.hist(model.theta, 50, facecolor='g', alpha=0.75)
    ax3.set_title('params histogram')
    plt.legend(loc='upper right')
    plt.style.use('ggplot')
    plt.savefig(f"./figures/Linearfig{n}.png")
    plt.show()
