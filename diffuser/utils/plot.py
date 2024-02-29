import numpy as np

def plot_losses(losses, ax, title='Losses'):
    steps = np.array([losses['training_losses'][i][0] for i in range(len(losses['training_losses']))])
    train_losses = np.array([losses['training_losses'][i][1].cpu().detach().numpy() for i in range(len(losses['training_losses']))])
    test_losses = np.array([losses['test_losses'][i][1] for i in range(len(losses['test_losses']))])
    ax.plot(steps, train_losses, label='train')
    ax.plot(steps, test_losses, label='test')
    ax.set_title(title)
    ax.legend()
    ax.set_ylim([0, max(max(train_losses[5:]), max(test_losses[5:]))])