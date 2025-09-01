import torch.distributions as tdist
import torch
import numpy as np
from scipy.ndimage import uniform_filter1d
import os
import pickle
from tqdm import tqdm
from utils.data_utils import plot_histogram, plot_distributions, plot_forward_pass, plot_latent_interpolation, plot_averaged_results

def calculate_log_likelihood(dec_mean_t_, dec_std_t_, Sx_t_):
    dec_mean_t_ = dec_mean_t_.to(Sx_t_.device)
    dec_std_t_ = dec_std_t_.to(Sx_t_.device)
    pred_dist = tdist.Normal(dec_mean_t_, dec_std_t_)
    log_probs = pred_dist.log_prob(Sx_t_)
    log_likelihoods = log_probs.sum(dim=[1, 2])
    return log_likelihoods.cpu().numpy()


def interpolate_latent(z_p, z_q, num_steps=10):
    interpolated_tensors = []
    for alpha in torch.linspace(0, 1, num_steps):
        interpolated_t = ((1 - alpha.item()) * z_p) + (alpha.item() * z_q)
        interpolated_tensors.append(interpolated_t)
    return interpolated_tensors


def calculate_vaf(y_true, y_pred):
    """
    Calculate the Variance Accounted For (VAF) for a batch of data.

    Args:
        y_true (torch.Tensor): Actual data with shape (batch_size, Channel_dim, Length)
        y_pred (torch.Tensor): Predicted data with shape (batch_size, Channel_dim, Length)

    Returns:
        torch.Tensor: VAF for each channel averaged over the batch (shape: Channel_dim)
    """
    assert y_true.shape == y_pred.shape
    residuals = y_true - y_pred
    var_y_true = torch.var(y_true, dim=-1, unbiased=False)  # shape: (batch_size, Channel_dim)
    var_residuals = torch.var(residuals, dim=-1, unbiased=False)  # shape: (batch_size, Channel_dim)
    vaf = 1 - (var_residuals / var_y_true)  # shape: (batch_size, Channel_dim)
    vaf_mean = torch.mean(vaf, dim=0)  # shape: (Channel_dim,)
    vaf_percentage = vaf_mean * 100
    return vaf_percentage, vaf


class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0.09):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0



# ------------------------------------------------------------------------------------------------------------------------------------------
# Testing Methods
# ------------------------------------------------------------------------------------------------------------------------------------------
def seqvae_mse_test(model, device, dataloader_er, save_dir=None, tag="_"):
    # Use lists to accumulate data for efficiency
    mse_all_list = []
    mse_energy_norm_list = []
    vaf_all_list = []
    log_likelihood_list = []
    st_list = []
    snr_all_list = []

    model.to(device)
    model.eval()
    with torch.no_grad():
        for j, complete_batched_data_t in tqdm(enumerate(dataloader_er),
                                               total=len(dataloader_er)):
            batched_data_t = complete_batched_data_t[0].to(device)
            results_t = model(batched_data_t)
            dec_mean_t_ = results_t.decoder_mean[:, :, 20:280]  # (batch, input_dim, length)
            dec_std_t_ = torch.sqrt(torch.exp(results_t.decoder_std))[:, :, 20:280]
            Sx_t_ = results_t.sx.permute(1, 2, 0)[:, :, 20:280]  # (batch, input_dim, length)
            # MSE per channel
            mse_per_coeff = torch.mean((Sx_t_ - dec_mean_t_) ** 2, dim=2)  # (batch, input_dim)
            # Energy of the original signal
            energy_per_coeff = torch.mean(Sx_t_ ** 2, dim=2)  # (batch, input_dim)
            # Energy-normalized MSE
            energy_normalized_mse = mse_per_coeff / (energy_per_coeff + 1e-12)
            # VAF calculation
            _, vaf = calculate_vaf(Sx_t_, dec_mean_t_)  # (input_dim,)
            # vaf = vaf.unsqueeze(0)  # make it (1, input_dim) for concatenation
            # Log-likelihood calculation
            log_likelihoods = calculate_log_likelihood(dec_mean_t_, dec_std_t_, Sx_t_)
            # SNR calculation (in dB)
            signal_power = torch.mean(Sx_t_ ** 2, dim=2)  # (batch, input_dim)
            noise_power = torch.mean((Sx_t_ - dec_mean_t_) ** 2, dim=2)  # (batch, input_dim)
            snr = 10.0 * torch.log10((signal_power + 1e-12) / (noise_power + 1e-12))  # (batch, input_dim)
            # Accumulate results
            mse_all_list.append(mse_per_coeff)
            mse_energy_norm_list.append(energy_normalized_mse)
            vaf_all_list.append(vaf)
            log_likelihood_list.extend(log_likelihoods)
            st_list.append(Sx_t_)
            snr_all_list.append(snr)

    tag_hist = tag + 'loglikelihood_'
    save_dir_hist = os.path.join(save_dir, tag_hist)
    os.makedirs(save_dir_hist, exist_ok=True)
    # Concatenate all data
    mse_all_data = torch.cat(mse_all_list, dim=0)  # (N, input_dim)
    mse_energy_normalized = torch.cat(mse_energy_norm_list, dim=0)  # (N, input_dim)
    vaf_all_data = torch.cat(vaf_all_list, dim=0)  # (N, input_dim)
    all_st_tensor = torch.cat(st_list, dim=0)  # (N, input_dim, length)
    snr_all_data = torch.cat(snr_all_list, dim=0)  # (N, input_dim)
    save_path_snr = os.path.join(save_dir_hist, f'{tag}-snr.npy')
    np.save(save_path_snr, snr_all_data.detach().cpu().numpy())
    # Mean and std of the entire dataset
    all_st_mean = all_st_tensor.mean(dim=0)  # (input_dim, length)
    all_st_std = all_st_tensor.std(dim=0)  # (input_dim, length)

    # Plot distributions of Sx
    plot_distributions(
        sx_mean=all_st_mean.detach().cpu().numpy(),
        sx_std=all_st_std.detach().cpu().numpy(),
        plot_second_channel=False,
        plot_sample=False,
        plot_dir=save_dir_hist,
        plot_dataset_average=True,
        tag='st_mean'
    )
    # Plot histogram of log-likelihood
    plot_histogram(
        data=np.array(log_likelihood_list),
        single_channel=True,
        bins=160,
        save_dir=save_dir_hist,
        tag='loglikelihood_original'
    )
    # Save VAF data
    vaf_path = os.path.join(save_dir_hist, f'{tag}-vaf_all_data_all_channels.npy')
    np.save(vaf_path, vaf_all_data.detach().cpu().numpy())
    # Averages across channels for MSE
    mse_all_data_averaged = torch.mean(mse_all_data, dim=1)  # (N,)
    mse_energy_normalized_averaged = torch.mean(mse_energy_normalized, dim=1)  # (N,)

    # Save MSE averaged data
    mse_avg_path = os.path.join(save_dir_hist, f'{tag}-mse_all_data_averaged.npy')
    np.save(mse_avg_path, mse_all_data_averaged.detach().cpu().numpy())
    mse_norm_avg_path = os.path.join(save_dir_hist, f'{tag}-mse_all_data_normalized_averaged.npy')
    np.save(mse_norm_avg_path, mse_energy_normalized_averaged.detach().cpu().numpy())
    # Plot histograms for MSE distributions
    plot_histogram(
        data=mse_all_data_averaged.detach().cpu().numpy(),
        single_channel=True,
        bins=160,
        save_dir=save_dir_hist,
        tag='mse-all_dist'
    )
    plot_histogram(
        data=mse_all_data.detach().cpu().numpy(),
        single_channel=False,
        bins=160,
        save_dir=save_dir_hist,
        tag='mse-all-data-per'
    )
    # SNR averaged per sample
    snr_all_data_averaged = torch.mean(snr_all_data, dim=1)  # (N,)
    snr_hist_path = os.path.join(save_dir_hist, f'{tag}-snr_all_data.npy')
    np.save(snr_hist_path, snr_all_data.detach().cpu().numpy())
    # Plot SNR histogram for all data (per-channel)
    plot_histogram(
        data=snr_all_data.detach().cpu().numpy(),
        single_channel=False,
        bins=160,
        save_dir=save_dir_hist,
        tag='snr-all-data-per'
    )
    # Plot SNR histogram averaged over channels (similar to mse-all_dist)
    plot_histogram(
        data=snr_all_data_averaged.detach().cpu().numpy(),
        single_channel=True,
        bins=160,
        save_dir=save_dir_hist,
        tag='snr-all_dist'
    )
    # VAF averaged per sample
    vaf_all_data_averaged = torch.mean(vaf_all_data, dim=1)  # (N,)
    vaf_hist_path = os.path.join(save_dir_hist, f'{tag}-vaf_all_data_all_channels_averaged.npy')
    np.save(vaf_hist_path, vaf_all_data_averaged.detach().cpu().numpy())
    # Plot VAF histogram for all data (per-channel)
    plot_histogram(
        data=vaf_all_data.detach().cpu().numpy(),
        single_channel=False,
        bins=160,
        save_dir=save_dir_hist,
        tag='vaf-all-data-per'
    )
    # Plot VAF histogram averaged over channels (similar to mse-all_dist)
    plot_histogram(
        data=vaf_all_data_averaged.detach().cpu().numpy(),
        single_channel=True,
        bins=160,
        save_dir=save_dir_hist,
        tag='vaf-all_dist'
    )

def plot_seqvae_tests(dataloader_pst, model, input_dim_t, device, base_dir=None):
    model.to(device)
    model.eval()
    mse_all_data = torch.empty((0, input_dim_t)).to(device)
    log_likelihood_all_data = []
    all_st = []
    with torch.no_grad():
        for j, complete_batched_data_t in tqdm(enumerate(dataloader_pst),
                                               total=len(dataloader_pst)):
            batched_data_t = complete_batched_data_t[0]
            # batched_data_t[:, [0, 1], :] = batched_data_t[:, [1, 0], :]
            guids = complete_batched_data_t[1]
            time_before_delivery = complete_batched_data_t[2]
            batched_data_t = batched_data_t.to(device)  # (batch_size, signal_len)
            results_t = model(batched_data_t)
            z_latent_t_ = results_t.z_latent  # (batch_size, latent_dim, 150)
            # h_hidden_t_ = results_t.hidden_states  # (hidden_layers, batch_size, input_len, h_dim)
            # if h_hidden_t_.dim() == 4:
            #     h_hidden_t__ = h_hidden_t_[-1].permute(0, 2, 1)
            # else:
            #     h_hidden_t__ = h_hidden_t_.permute(0, 2, 1)
            dec_mean_t_ = results_t.decoder_mean  # (batch_size, input_dim, input_size)
            # dec_std_t_ = torch.sqrt(torch.exp(results_t.decoder_std))
            Sx_t_ = results_t.sx.permute(1, 2, 0)  # (batch_size, input_dim, 150)
            # enc_mean_t_ = results_t.encoder_mean  # (batch_size, input_dim, 150)
            # enc_std_t_ = torch.sqrt(torch.exp(results_t.encoder_std))
            # kld_values_t_ = results_t.kld_values

            mse_per_coefficients = torch.sum(((Sx_t_ - dec_mean_t_) ** 2), dim=2) / Sx_t_.size(-1)
            mse_all_data = torch.cat((mse_all_data, mse_per_coefficients), dim=0)
            # log_likelihoods = calculate_log_likelihood(dec_mean_t_, dec_std_t_, Sx_t_)
            # log_likelihood_all_data.extend(log_likelihoods)
            all_st.append(Sx_t_)
            save_dir = os.path.join(base_dir, 'Complete seqvae testing')
            os.makedirs(save_dir, exist_ok=True)
            signal_channel_dim = Sx_t_.shape[1]
            signal_len = Sx_t_.shape[2]
            selected_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            # if j==2:
            for signal_index in selected_idx:
                save_dir_signal = save_dir
                selected_signal = batched_data_t[signal_index]
                sx_selected = Sx_t_[signal_index][:, 16:280]  # (input_dim, input_size)
                z_selected = z_latent_t_[signal_index]
                # input_data_for_tsne = sx_selected.permute(1, 0).detach().cpu().numpy()
                # latent_data_for_tsne = z_selected.permute(1, 0).detach().cpu().numpy()
                # tsne = TSNE(n_components=2, random_state=42)
                # input_tsne_results = tsne.fit_transform(input_data_for_tsne)
                # latent_tsne_results = tsne.fit_transform(latent_data_for_tsne)
                # fig, ax = plt.subplots(nrows=2, figsize=(6, 2 * 6 + 3))
                # ax[0].scatter(input_tsne_results[:, 0], input_tsne_results[:, 1],
                #               c=np.linspace(0, 1, signal_len), cmap='Blues', s=100, edgecolors='black')
                # ax[0].set_ylabel('st original')
                #
                # ax[1].scatter(latent_tsne_results[:, 0], latent_tsne_results[:, 1],
                #               c=np.linspace(0, 1, signal_len), cmap='Reds', s=100, edgecolors='black')
                # ax[1].set_ylabel('latent representation')
                # plt.savefig(save_dir_signal + '/' + 't-SNE' + '.pdf', bbox_inches='tight',
                #             orientation='landscape',
                #             dpi=50)
                # plt.close(fig)

                # if channel_num == 1:
                #     signal_c = selected_signal.detach().cpu().numpy()  # for 1 channel
                #     two_channel_flag = False
                # else:
                #     signal_c = selected_signal.squeeze(0).permute(1, 0).detach().cpu().numpy()  # for 2 channel
                #     two_channel_flag = False
                # plot_averaged_results(signal=signal_c, Sx=sx_selected.detach().cpu().numpy(),
                #                       Sxr_mean=dec_mean_t_[signal_index].detach().cpu().numpy(),
                #                       Sxr_std=dec_std_t_[signal_index].detach().cpu().numpy(),
                #                       z_latent_mean=enc_mean_t_[signal_index].detach().cpu().numpy(),
                #                       z_latent_std=enc_std_t_[signal_index].detach().cpu().numpy(),
                #                       kld_values=kld_values_t_[signal_index].detach().cpu().numpy(),
                #                       h_hidden_mean=h_hidden_t__[signal_index].detach().cpu().numpy(),
                #                       plot_latent=True,
                #                       plot_klds=True,
                #                       two_channel=two_channel_flag,
                #                       plot_state=False,
                #                       # new_sample=new_sample.detach().cpu().numpy(),
                #                       plot_dir=save_dir_signal, tag=f'-{signal_index}')
                plot_forward_pass(signal=selected_signal.detach().cpu().numpy(),
                                  plot_title=f'{guids[signal_index]}-'
                                              f'{time_before_delivery[signal_index].detach().cpu()}',
                                  plot_second_channel=(self.input_channel_num == 2),
                                  fhr_st=sx_selected.detach().cpu().numpy(), meta=None,
                                  fhr_st_pr=dec_mean_t_[signal_index][:, 16:280].detach().cpu().numpy(),
                                  Sxr_std=results_t.decoder_std[signal_index][:, 16:280].detach().cpu().numpy(),
                                  z_latent=z_latent_t_[signal_index].detach().cpu().numpy(),
                                  plot_dir=save_dir_signal, tag=f'-{signal_index}-{j}')
