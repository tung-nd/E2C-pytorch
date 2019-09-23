import torch
import torchvision
from tensorboardX import SummaryWriter
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import numpy as np
import matplotlib.pyplot as plt
import argparse

from normal import *
from e2c_model import E2C
from datasets import *
import data.sample_planar_data as planar_sampler
import data.sample_pendulum_data as pendulum_sampler
import data.sample_cartpole_data as cartpole_sampler

np.random.seed(0)
torch.manual_seed(0)
torch.set_default_dtype(torch.float64)

device = torch.device("cuda")
datasets = {'planar': PlanarDataset, 'pendulum': GymPendulumDatasetV2}
settings = {'planar': (1600, 2, 2), 'pendulum': (4608, 3, 1)}
samplers = {'planar': planar_sampler, 'pendulum': pendulum_sampler, 'cartpole': cartpole_sampler}
num_eval = 5 # number of images evaluated on tensorboard

# x, u, x_next = dataset[0]
# imgplot = plt.imshow(x.squeeze(), cmap='gray')
# plt.show()
# print (np.array(u, dtype=float))
# imgplot = plt.imshow(x_next.squeeze(), cmap='gray')
# plt.show()

def compute_loss(x, x_next, q_z_next, x_recon, x_next_pred, q_z, q_z_next_pred, lamda):
    # lower-bound loss
    recon_term = -torch.mean(torch.sum(x * torch.log(1e-5 + x_recon)
                                       + (1 - x) * torch.log(1e-5 + 1 - x_recon), dim=1))
    pred_loss = -torch.mean(torch.sum(x_next * torch.log(1e-5 + x_next_pred)
                                      + (1 - x_next) * torch.log(1e-5 + 1 - x_next_pred), dim=1))

    kl_term = - 0.5 * torch.mean(torch.sum(1 + q_z.logvar - q_z.mean.pow(2) - q_z.logvar.exp(), dim = 1))

    lower_bound = recon_term + pred_loss + kl_term

    # consistency loss
    consis_term = NormalDistribution.KL_divergence(q_z_next_pred, q_z_next)
    return lower_bound + lamda * consis_term

def train(model, train_loader, lam, optimizer):
    model.train()
    avg_loss = 0.0

    num_batches = len(train_loader)
    for i, (x, u, x_next) in enumerate(train_loader, 0):
        x = x.view(-1, model.obs_dim).double().to(device)
        u = u.double().to(device)
        x_next = x_next.view(-1, model.obs_dim).double().to(device)
        optimizer.zero_grad()

        x_recon, x_next_pred, q_z, q_z_next_pred, q_z_next = model(x, u, x_next)

        loss = compute_loss(x, x_next, q_z_next, x_recon, x_next_pred, q_z, q_z_next_pred, lam)

        avg_loss += loss.item()
        loss.backward()
        optimizer.step()

    return avg_loss / num_batches

def compute_log_likelihood(x, x_recon, x_next, x_next_pred):
    loss_1 = -torch.mean(torch.sum(x * torch.log(1e-5 + x_recon)
                                   + (1 - x) * torch.log(1e-5 + 1 - x_recon), dim=1))
    loss_2 = -torch.mean(torch.sum(x_next * torch.log(1e-5 + x_next_pred)
                                   + (1 - x_next) * torch.log(1e-5 + 1 - x_next_pred), dim=1))
    return loss_1, loss_2

def evaluate(model, test_loader):
    model.eval()
    num_batches = len(test_loader)
    state_loss, next_state_loss = 0., 0.
    with torch.no_grad():
        for x, u, x_next in test_loader:
            x = x.view(-1, model.obs_dim).double().to(device)
            u = u.double().to(device)
            x_next = x_next.view(-1, model.obs_dim).double().to(device)

            x_recon, x_next_pred, q_z, q_z_next_pred, q_z_next = model(x, u, x_next)
            loss_1, loss_2 = compute_log_likelihood(x, x_recon, x_next, x_next_pred)
            state_loss += loss_1
            next_state_loss += loss_2

    return state_loss.item() / num_batches, next_state_loss.item() / num_batches

# code for visualizing the training process
def predict_x_next(model, env, num_eval):
    # frist sample a true trajectory from the environment
    sampler = samplers[env]
    sampled_data = sampler.sample_for_eval(num_eval)

    # use the trained model to predict the next observation
    predicted = []
    for x, u, x_next in sampled_data:
        x_reshaped = x.reshape(-1)
        x_reshaped = torch.from_numpy(x_reshaped).double().unsqueeze(dim=0).to(device)
        u = torch.from_numpy(u).double().unsqueeze(dim=0).to(device)
        with torch.no_grad():
            x_next_pred = model.predict(x_reshaped, u)
        predicted.append(x_next_pred.squeeze().cpu().numpy().reshape(sampler.width, sampler.height))
    true_x_next = [data[-1] for data in sampled_data]
    return true_x_next, predicted

def plot_preds(model, env, num_eval):
    true_x_next, pred_x_next = predict_x_next(model, env, num_eval)

    # plot the predicted and true observations
    fig, axes =plt.subplots(nrows=2, ncols=num_eval)
    pad = 5
    axes[0, 0].annotate('True observations', xy=(0, 0.5), xytext=(-axes[0,0].yaxis.labelpad - pad, 0),
                   xycoords=axes[0,0].yaxis.label, textcoords='offset points',
                   size='large', ha='right', va='center')
    axes[1, 0].annotate('Predicted observations', xy=(0, 0.5), xytext=(-axes[1, 0].yaxis.labelpad - pad, 0),
                        xycoords=axes[1, 0].yaxis.label, textcoords='offset points',
                        size='large', ha='right', va='center')

    for idx in np.arange(num_eval):
        axes[0, idx].imshow(true_x_next[idx], cmap='Greys')
        axes[1, idx].imshow(pred_x_next[idx], cmap='Greys')
    fig.tight_layout()
    return fig

def main(args):
    env_name = args.env
    assert env_name in ['planar', 'pendulum']
    propor = args.propor
    batch_size = args.batch_size
    lr = args.lr
    lam = args.lam
    epoches = args.num_iter
    iter_save = args.iter_save

    dataset = datasets[env_name]('./data/data/' + env_name)
    train_set, test_set = dataset[:int(len(dataset) * propor)], dataset[int(len(dataset) * propor):]
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=8)

    obs_dim, z_dim, u_dim = settings[env_name]
    model = E2C(obs_dim=obs_dim, z_dim=z_dim, u_dim=u_dim, env=env_name).to(device)

    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.9), eps=1e-8, lr=lr)

    writer = SummaryWriter('logs/planar')

    for i in range(epoches):
        avg_loss = train(model, train_loader, lam, optimizer)
        print('Epoch %d' % i)
        print("Training loss: %f" % (avg_loss))
        # evaluate on test set
        state_loss, next_state_loss = evaluate(model, test_loader)
        print('State loss: ' + str(state_loss))
        print('Next state loss: ' + str(next_state_loss))

        # ...log the running loss
        writer.add_scalar('training loss', avg_loss, i)
        writer.add_scalar('state loss', state_loss, i)
        writer.add_scalar('next state loss', next_state_loss, i)

        # save model
        if (i + 1) % iter_save == 0:
            writer.add_figure('actual vs. predicted observations',
                              plot_preds(model, env_name, 5),
                              global_step=i)
            print('Saving the model.............')
            # torch.save(model, './model_planar_' + str(i + 1))
            if not path.exists('./result/' + env_name):
                os.makedirs('./result/' + env_name)
            torch.save(model.state_dict(), './result/' + env_name + '/model_' + str(i + 1))
            with open('./result/' + env_name + '/loss_' + str(i + 1), 'w') as f:
                f.write('\n'.join([str(state_loss), str(next_state_loss)]))

    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train e2c model')

    # the default value is used for the planar task
    parser.add_argument('--env', required=True, type=str, help='the environment used for training')
    parser.add_argument('--propor', default=3/4, type=float, help='the proportion of data used for training')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--lr', default=0.0001, type=float, help='the learning rate')
    parser.add_argument('--lam', default=0.25, type=float, help='the weight of the consistency term')
    parser.add_argument('--num_iter', default=5000, type=int, help='the number of epoches')
    parser.add_argument('--iter_save', default=1000, type=int, help='save model and result after this number of iterations')

    args = parser.parse_args()

    main(args)