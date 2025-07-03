import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import torch
import torch.nn as nn
import torchvision
import numpy as np
from matplotlib.colors import LogNorm
import matplotlib as mpl
from kymatio.torch import Scattering1D
import pickle
from scipy.signal import decimate
import plotly.graph_objects as go
import os
import matplotlib.animation as animation
import matplotlib.colors as mcolors


def calculate_stats(loader):
    mean = 0.0
    std = 0.0
    total_samples = 0

    for data in loader:
        # Assuming data shape is [batch_size, channels, ...]
        data = data.unsqueeze(1)
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        total_samples += batch_samples

    mean /= total_samples
    std /= total_samples

    return mean, std


def normalize_data(seq_list, min_val, max_val):
    # todo there could be a better way to do normalization in the pipline
    range_val = max_val - min_val
    for dict_item in seq_list:
        dict_item['fhr'] = [((x - min_val) / range_val) for x in dict_item['fhr']]


def prepare_data(file_path=None, do_decimate=True):
    with open(file_path, 'rb') as input_file:
        dict_list = pickle.load(input_file)
    if do_decimate:
        for dict_item in dict_list:
            dict_item['fhr'] = decimate(dict_item['fhr'], 16).tolist()
    return dict_list


def plot_scattering(signal=None, plot_order=None, Sx=None, meta=None,
                    Sxr=None, plot_dir=None, tag=''):
    if torch.is_tensor(signal) and signal.is_cuda:
        signal = signal.cpu().detach().numpy()
    if torch.is_tensor(Sx) and Sx.is_cuda:
        Sx = Sx.cpu().detach().numpy()
    Fs = 4
    Q = 1
    J = 11
    Over = 0
    T = 2 ** (J - 7)
    N_CHAN = 12
    log_eps = 1e-3
    dtype = np.float32
    SINGLE_BATCH_SIZE = 1
    N = len(signal)
    if Sxr is not None:
        # N_ROWS = 3
        N_ROWS = len(plot_order) + 4
    else:
        # N_ROWS = 2
        N_ROWS = len(plot_order) + 1

    t_in = np.arange(0, N) / Fs

    cmstr = 'Blues'
    plt.set_cmap(cmstr)
    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 8, 'axes.labelsize': 8})
    i_row = 0

    fig, ax = plt.subplots(nrows=N_ROWS, ncols=2, figsize=(14, 16),
                           gridspec_kw={"width_ratios": [40, 1]})
    ax[i_row, 1].set_axis_off()
    ax[i_row, 0].plot(t_in, signal, linewidth=1.5)
    ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
    ax[i_row, 0].set_xticklabels([])
    ax[i_row, 0].set_ylabel('FHR (bpm)')


    for order in plot_order:
        if isinstance(order, int):
            i_row += 1
            order_i = np.where(meta['order'] == order)
            x = Sx[:, order_i, :].squeeze()
            if order == 0:
                ax[i_row, 0].plot(x.squeeze(), linewidth=1.5)
                ax[i_row, 1].set_axis_off()
            else:
                imgplot = ax[i_row, 0].imshow(np.log(x + log_eps), aspect='auto',
                                              extent=[0, N / Fs, Sx.shape[0], 0])
                # imgplot = ax[i_row, 0].imshow(x, aspect='auto')
                ax[i_row, 1].set_axis_on()
                fig.colorbar(imgplot, cax=ax[i_row, 1])
            ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
            ax[i_row, 0].set_xticklabels([])
            ax[i_row, 0].set_ylabel(f'Order {order}')
        elif isinstance(order, tuple):
            i_row += 1
            order_i = np.where(np.isin(meta['order'], order))
            x = Sx[:, order_i, :].squeeze()
            imgplot = ax[i_row, 0].imshow(np.log(x + log_eps), aspect='auto',
                                          extent=[0, N / Fs, Sx.shape[0], 0])
            # imgplot = ax[i_row, 0].imshow(x, aspect='auto')
            ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
            ax[i_row, 0].set_xticklabels([])
            ax[i_row, 0].set_ylabel(f'Order {order}')
            ax[i_row, 1].set_axis_on()
            fig.colorbar(imgplot, cax=ax[i_row, 1])

    if Sxr is not None:
        i_row += 1
        if torch.is_tensor(Sx) and Sxr.is_cuda:
            Sxr = Sxr.cpu().detach().numpy()
        Sxr = Sxr.transpose(1, 0)

        ax[i_row, 0].plot(Sxr[0, :], linewidth=1.5)
        ax[i_row, 1].set_axis_off()
        ax[i_row, 0].autoscale(enable=True, axis='x reconstructed', tight=True)
        ax[i_row, 0].set_xticklabels([])
        ax[i_row, 0].set_ylabel('Reconstructed order 0')

        i_row += 1
        imgplot = ax[i_row, 0].imshow(np.log(Sxr[1:, :] + log_eps), aspect='auto',
                                      extent=[0, N / Fs, Sxr.shape[0], 0])
        ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
        ax[i_row, 0].set_xticklabels([])
        ax[i_row, 0].set_ylabel('Reconstructed order 1')
        ax[i_row, 1].set_axis_on()
        fig.colorbar(imgplot, cax=ax[i_row, 1])

        i_row += 1
        imgplot = ax[i_row, 0].imshow(np.log(Sxr + log_eps), aspect='auto',
                                      extent=[0, N / Fs, Sxr.shape[0], 0])
        ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
        ax[i_row, 0].set_xticklabels([])
        ax[i_row, 0].set_ylabel('Reconstructed order 1 and 0')
        ax[i_row, 1].set_axis_on()
        fig.colorbar(imgplot, cax=ax[i_row, 1])

    cmstr = 'bwr'
    plt.set_cmap(cmstr)
    fig.delaxes(ax[1][1])
    ax[0, 1].set_axis_off()
    # plt.savefig(plot_dir + '/' + record_name + '_' + str(domain_start[i_segment]) + '_st.pdf', bbox_inches='tight',
    #             orientation='landscape')
    plt.savefig(plot_dir + '/' + tag + '_' + '.pdf', bbox_inches='tight', orientation='landscape', dpi=50)
    plt.close(fig)


def plot_original_reconstructed(original_x, reconstructed_x, plot_dir=None, tag=''):
    # Set the font globally to Times New Roman, size 18
    # mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['font.size'] = 14

    fig, ax = plt.subplots(3, 1, figsize=(30, 7))
    ax[0].plot(original_x, label='Original')
    ax[1].plot(reconstructed_x, label='Reconstructed')
    ax[2].plot(original_x, label='Original', linewidth=2, color='#474747')
    ax[2].plot(reconstructed_x, label='Reconstructed', linewidth=1.5, color='#43AA8B')

    # Adding legends
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()

    # Setting x and y labels
    ax[0].set_xlabel('Sample')
    ax[0].set_ylabel('FHR')
    ax[1].set_xlabel('Sample')
    ax[1].set_ylabel('Reconstructed FHR')

    # Showing grid
    ax[0].grid(True)
    ax[1].grid(True)
    ax[2].grid(True)
    plt.savefig(plot_dir + '/' + tag + '_' + '_st.pdf', bbox_inches='tight', orientation='landscape')
    plt.close(fig)

def plot_original_reconstructed_vae(raw_signal, original_x, reconstructed_x, plot_dir=None, tag=''):
    # Set the font globally to Times New Roman, size 18
    # mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['font.size'] = 14

    fig, ax = plt.subplots(3, 1, figsize=(30, 7))
    ax[0].plot(raw_signal, label='Original', linewidth=3)
    ax[0].plot(original_x, label='st_rec',  linewidth=1.5)

    ax[1].plot(raw_signal, label='Original', linewidth=3)
    ax[1].plot(reconstructed_x, label='rec_st_rec', linewidth=1.5)

    ax[2].plot(raw_signal, label='Original', linewidth=2.5, color='#474747')
    ax[2].plot(original_x, label='st_rec', linewidth=2, color='#43AA8B')
    ax[2].plot(reconstructed_x, label='rec_st_rec', linewidth=1.5, color='#C7253E')

    # Adding legends
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()

    # Setting x and y labels
    ax[0].set_xlabel('Sample')
    ax[0].set_ylabel('FHR')
    ax[1].set_xlabel('Sample')
    ax[1].set_ylabel('Reconstructed FHR')

    # Showing grid
    ax[0].grid(True)
    ax[1].grid(True)
    ax[2].grid(True)
    plt.savefig(plot_dir + '/' + tag + '.pdf', bbox_inches='tight', orientation='landscape')
    plt.close(fig)

def plot_prediction_st(signal=None, plot_title='', sx=None, plot_second_channel=False, tag='', sx_pmean=None,
                       sx_pvar=None, plot_dir=None, prediction_idx=None, vline_indices=None):
    Fs=4
    if plot_second_channel:
        n_rows = 3 + sx.shape[0]
        signal_1 = signal[0,:]
        signal_2 = signal[1,:]
    else:
        n_rows = 2 + sx.shape[0]
        signal_1 = signal[:,0]
        signal_2 = signal_1
    N = len(signal_1)
    t_in = np.arange(0, N) / Fs
    cmstr = 'seismic'
    plt.set_cmap(cmstr)
    plt.rcParams.update({'font.size': 19, 'axes.titlesize': 18, 'axes.labelsize': 18})
    i_row = 0
    fig, ax = plt.subplots(nrows=n_rows, ncols=2, figsize=(25, n_rows * 4 + 10),
                           gridspec_kw={"width_ratios": [80, 1]})
    ax[i_row, 1].set_axis_off()
    ax[i_row, 0].plot(t_in, signal_1, linewidth=1.5)
    ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
    ax[i_row, 0].set_xticklabels([])
    ax[i_row, 0].set_ylabel('FHR (bpm)')

    if plot_second_channel:
        i_row += 1
        ax[i_row, 1].set_axis_off()
        ax[i_row, 0].plot(t_in, signal_2, linewidth=1.5)
        ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
        ax[i_row, 0].set_xticklabels([])
        ax[i_row, 0].set_ylabel('UP')
    i_row += 1
    imgplot = ax[i_row, 0].imshow(sx, aspect='auto', norm="symlog",
                                  extent=[0, N / Fs, sx.shape[0], 0], interpolation='none')
    ax[i_row, 1].set_axis_on()
    fig.colorbar(imgplot, cax=ax[i_row, 1])
    ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
    ax[i_row, 0].set_xticklabels([])
    ax[i_row, 0].set_ylabel('True ST')
    x_values_for_pmean = range(prediction_idx, prediction_idx + len(sx_pmean[0, :]))
    for i in range(sx.shape[0]):
        i_row += 1
        ax[i_row, 0].plot(sx[i, :], linewidth=3, color="#034C53", label="True")
        ax[i_row, 0].plot(x_values_for_pmean, sx_pmean[i, :], linewidth=2.5, color="#FA4032", label="Predicted")
        if sx_pvar is not None:
            ax[i_row, 0].fill_between(x_values_for_pmean,
                                      sx_pmean[i, :] - 0.7*sx_pvar[i, :],
                                      sx_pmean[i, :] + 0.7*sx_pvar[i, :],
                                      color='blue', alpha=0.1, label='Std dev')

        if vline_indices is not None:
            for idx in vline_indices:
                ax[i_row, 0].axvline(x=idx, color='#80CBC4', linestyle=(0, (1, 1)), linewidth=3)

        ax[i_row, 0].legend()
        ax[i_row, 1].set_axis_off()
        ax[i_row, 0].set_ylabel(f'Coefficient {i}')

    fig.delaxes(ax[1][1])
    ax[0, 1].set_axis_off()
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.suptitle(plot_title, fontsize=22)
    plt.savefig(plot_dir + '/st-plots' + tag + '.pdf', bbox_inches='tight', orientation='landscape', dpi=50)
    plt.close(fig)

def plot_forward_pass(raw_fhr=None, raw_up=None, fhr_st=None, fhr_ph=None,
                      fhr_st_mean_pred=None, fhr_ph_mean_pred=None,
                      z_latent=None,
                      plot_dir=None, plot_title='', tag=''):
    """
    Plots a comprehensive view of the model's forward pass for the TEB VAE.

    Args:
        raw_fhr (np.ndarray): Raw FHR signal.
        raw_up (np.ndarray): Raw UP signal.
        fhr_st (np.ndarray): Ground truth scattering transform of FHR. Shape (time, channels)
        fhr_ph (np.ndarray): Ground truth phase harmonics of FHR. Shape (time, channels)
        fhr_st_mean_pred (np.ndarray): Predicted mean of scattering transform. Shape (time, channels)
        fhr_ph_mean_pred (np.ndarray): Predicted mean of phase harmonics. Shape (time, channels)
        z_latent (np.ndarray): Latent space representation. Shape (latent_dim, time)
        plot_dir (str): Directory to save the plot.
        plot_title (str): Title for the plot.
        tag (str): A unique tag for the output filename.
    """
    Fs = 4
    num_coeffs_to_plot = 11

    # Transpose coefficient matrices for plotting from (time, channels) to (channels, time)
    fhr_st_mean_pred = fhr_st_mean_pred.T
    fhr_ph_mean_pred = fhr_ph_mean_pred.T
    z_latent = z_latent.T

    N = len(raw_fhr)
    t_in = np.arange(0, N) / Fs

    # Setup plot styling
    plt.set_cmap('seismic')
    plt.rcParams.update({'font.size': 14, 'axes.titlesize': 16, 'axes.labelsize': 14})

    # Determine number of rows for the plot
    n_rows = 7 + num_coeffs_to_plot * 2
    fig, ax = plt.subplots(nrows=n_rows, ncols=2, figsize=(28, n_rows * 4),
                           gridspec_kw={"width_ratios": [80, 1]}, constrained_layout=True)

    i_row = 0

    # --- Plot Raw Signals ---
    # Raw FHR
    ax[i_row, 1].set_axis_off()
    ax[i_row, 0].plot(t_in, raw_fhr, linewidth=1.5, color='blue')
    ax[i_row, 0].set_ylabel('Raw FHR (bpm)')
    ax[i_row, 0].set_title('Raw Fetal Heart Rate')
    ax[i_row, 0].grid(True, linestyle='--', alpha=0.6)
    ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
    i_row += 1
    
    # Raw UP
    ax[i_row, 1].set_axis_off()
    ax[i_row, 0].plot(t_in, raw_up, linewidth=1.5, color='orange')
    ax[i_row, 0].set_ylabel('Raw UP')
    ax[i_row, 0].set_title('Raw Uterine Pressure')
    ax[i_row, 0].grid(True, linestyle='--', alpha=0.6)
    ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
    i_row += 1

    # --- Plot Latent Space ---
    if z_latent is not None:
        imgplot = ax[i_row, 0].imshow(z_latent, aspect='auto', cmap='viridis', origin='lower', interpolation='none')
        ax[i_row, 1].set_axis_on()
        fig.colorbar(imgplot, cax=ax[i_row, 1])
        ax[i_row, 0].set_ylabel('Latent Dim')
        ax[i_row, 0].set_title('Latent Space Representation (z)')
        i_row += 1
    else:
        # If z_latent is not provided, hide the row
        ax[i_row, 0].set_visible(False)
        ax[i_row, 1].set_visible(False)
        i_row += 1

    # --- Plot Coefficient Matrices (imshow) ---
    def plot_imshow(data, title, current_row):
        imgplot = ax[current_row, 0].imshow(data, aspect='auto', norm="symlog", interpolation='none')
        ax[current_row, 1].set_axis_on()
        fig.colorbar(imgplot, cax=ax[current_row, 1])
        ax[current_row, 0].set_ylabel('Coefficient Index')
        ax[current_row, 0].set_title(title)

    # ST plots
    plot_imshow(fhr_st, 'Ground Truth Scattering Coefficients (ST)', i_row)
    i_row += 1
    plot_imshow(fhr_st_mean_pred, 'Predicted Mean Scattering Coefficients (ST)', i_row)
    i_row += 1

    # Phase plots
    plot_imshow(fhr_ph, 'Ground Truth Phase Harmonics (PH)', i_row)
    i_row += 1
    plot_imshow(fhr_ph_mean_pred, 'Predicted Mean Phase Harmonics (PH)', i_row)
    i_row += 1


    # --- Plot Individual Coefficients (line plots) ---
    def plot_line_coeffs(true_coeffs, pred_coeffs, title_prefix, start_row, num_coeffs):
        # Plot title for the whole block of coefficients
        ax[start_row, 0].set_title(f'{title_prefix} Coefficients {0}-{num_coeffs-1}')
        for i in range(num_coeffs):
            ax[start_row + i, 0].plot(true_coeffs[i, :], linewidth=2.5, label="Ground Truth", color="#034C53")
            ax[start_row + i, 0].plot(pred_coeffs[i, :], linewidth=2, label="Predicted Mean", color="#FA4032", linestyle='--')
            ax[start_row + i, 0].set_ylabel(f'Coeff {i}')
            ax[start_row + i, 0].legend(loc='upper right')
            ax[start_row + i, 0].grid(True, linestyle='--', alpha=0.5)
            ax[start_row + i, 0].autoscale(enable=True, axis='x', tight=True)
            ax[start_row + i, 1].set_axis_off()
        return start_row + num_coeffs

    # Line plots for fhr_st
    i_row = plot_line_coeffs(fhr_st, fhr_st_mean_pred, 'ST', i_row, num_coeffs_to_plot)
    
    # Line plots for fhr_ph
    i_row = plot_line_coeffs(fhr_ph, fhr_ph_mean_pred, 'Phase', i_row, num_coeffs_to_plot)

    # Final adjustments
    fig.suptitle(plot_title, fontsize=24)
    
    # Save the plot
    save_path = os.path.join(plot_dir, f'forward_pass_{tag}.pdf')
    plt.savefig(save_path, bbox_inches='tight', orientation='landscape')
    plt.close(fig)


def plot_forward_pass_raw_signal(signal=None, plot_title='', Sx=None, plot_second_channel=False,
                                 fhr_pred_mean=None, fhr_pred_std=None, fhr_raw=None,
                                 z_latent=None, plot_dir=None, tag=''):
    Fs = 4
    log_eps = 1e-3
    N = len(signal)
    # if Sxr is not None:
    #     # N_ROWS = 3
    #     N_ROWS = len(plot_order) + 4
    # else:
    #     # N_ROWS = 2
    #     N_ROWS = len(plot_order) + 1
    if plot_second_channel:
        N_ROWS = 5
        signal_1 = signal[0, :]
        signal_2 = signal[1, :]
        N = len(signal_1)
    else:
        N_ROWS = 4
        signal_2 = signal
        signal_1 = signal
        N = len(signal_1)
    t_in = np.arange(0, N) / Fs

    cmstr = 'seismic'
    plt.set_cmap(cmstr)
    plt.rcParams.update({'font.size': 19, 'axes.titlesize': 18, 'axes.labelsize': 18})

    i_row = 0

    fig, ax = plt.subplots(nrows=N_ROWS, ncols=2, figsize=(25, N_ROWS * 4 + 10),
                           gridspec_kw={"width_ratios": [80, 1]})
    ax[i_row, 1].set_axis_off()
    ax[i_row, 0].plot(t_in, signal_1, linewidth=1.5)
    ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
    ax[i_row, 0].set_xticklabels([])
    ax[i_row, 0].set_ylabel('FHR (bpm)')

    if plot_second_channel:
        i_row += 1
        ax[i_row, 1].set_axis_off()
        ax[i_row, 0].plot(t_in, signal_2, linewidth=1.5)
        ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
        ax[i_row, 0].set_xticklabels([])
        ax[i_row, 0].set_ylabel('UP')

    i_row += 1
    imgplot = ax[i_row, 0].imshow(Sx, aspect='auto', norm="symlog",
                                  extent=[0, N / Fs, Sx.shape[0], 0], interpolation='none')
    ax[i_row, 1].set_axis_on()
    fig.colorbar(imgplot, cax=ax[i_row, 1])
    ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
    ax[i_row, 0].set_xticklabels([])
    ax[i_row, 0].set_ylabel('True ST')

    # i_row += 1
    # imgplot = ax[i_row, 0].imshow(Sxr, aspect='auto', norm="symlog",
    #                               extent=[0, N / Fs, Sx.shape[0], 0], interpolation='none')
    # ax[i_row, 1].set_axis_on()
    # fig.colorbar(imgplot, cax=ax[i_row, 1])
    # ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
    # ax[i_row, 0].set_xticklabels([])
    # ax[i_row, 0].set_ylabel('Reconstructed ST')

    i_row += 1
    imgplot = ax[i_row, 0].imshow(z_latent, aspect='auto', norm="linear",
                                  extent=[0, N / Fs, z_latent.shape[0], 0], interpolation='none')
    ax[i_row, 1].set_axis_on()
    fig.colorbar(imgplot, cax=ax[i_row, 1])
    ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
    ax[i_row, 0].set_xticklabels([])
    ax[i_row, 0].set_ylabel('Latent Representation')

    i_row += 1
    ax[i_row, 1].set_axis_off()
    ax[i_row, 0].plot(t_in, fhr_raw, linewidth=1.5, label="normalized raw", color="#16C47F")
    ax[i_row, 0].plot(t_in, fhr_pred_mean, linewidth=1.5, label="predicted", color="#E83F25")
    ax[i_row, 0].fill_between(fhr_pred_mean,
                              fhr_pred_mean - 0.7 * fhr_pred_std,
                              fhr_pred_mean - 0.7 * fhr_pred_std,
                              color='#EA7300', alpha=0.5, label='Std dev')
    ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
    ax[i_row, 0].set_xticklabels([])
    ax[i_row, 0].set_ylabel('FHR (bpm)')



    cmstr = 'bwr'
    plt.set_cmap(cmstr)
    fig.delaxes(ax[1][1])
    ax[0, 1].set_axis_off()
    # plt.savefig(plot_dir + '/' + record_name + '_' + str(domain_start[i_segment]) + '_st.pdf', bbox_inches='tight',
    #             orientation='landscape')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.suptitle(plot_title, fontsize=22)
    plt.savefig(plot_dir + '/st-plots' + tag + '.pdf', bbox_inches='tight', orientation='landscape', dpi=50)
    plt.close(fig)

def plot_loss_dict(loss_dict, epoch_num, plot_dir):
    # plt.rcParams["font.family"] = "Times New Roman"
    # plt.rcParams["font.size"] = 32
    # plt.rcParams['text.usetex'] = True
    num_rows = len(loss_dict.keys())
    t = np.arange(1, epoch_num + 1)
    # fig, ax = plt.subplots(nrows=num_rows, ncols=1, figsize=(15, 30))
    fig = go.Figure()
    for i, (key, val) in enumerate(loss_dict.items()):
        fig.add_trace(go.Scatter(y=val, mode='lines', name=key))
        # ax[i].autoscale(enable=True, axis='x', tight=True)
        # ax[i].plot(t, val, label=key, color='#265073', linewidth=0.7)
        # ax[i].set_ylabel(key, fontsize=14)
        # ax[i].grid()
    # fig = go.Figure()
    # Update layout to add titles and adjust other settings as needed
    fig.update_layout(title='Loss',
                      xaxis_title='Epoch',
                      yaxis_title='Loss',
                      legend_title='Legend',
                      template='plotly_dark')

    # Save the figure as an HTML file
    fig_path = os.path.join(plot_dir, 'loss_plot.html')
    fig.write_html(fig_path)
    # plt.savefig(f'{plot_dir}/Loss_st.pdf', bbox_inches='tight', dpi=50)


def plot_averaged_results(signal=None, Sx=None, Sxr_mean=None, Sxr_std=None, z_latent_mean=None, h_hidden_mean=None,
                          h_hidden_std=None, z_latent_std=None, kld_values=None, plot_dir=None, new_sample=None,
                          plot_latent=False, plot_klds=False, plot_state=False, two_channel=False, tag=''):
    Fs = 4
    log_eps = 1e-3

    if two_channel:
        signal_1 = signal[:, 0]
        signal_2 = signal[:, 1]
    else:
        signal_2 = signal
        signal_1 = signal

    N = len(signal_1)

    N_ROWS = 7 + (z_latent_mean.shape[0])
    t_in = np.arange(0, N) / Fs
    cmstr = 'seismic'
    # cmstr = 'Blues'
    plt.set_cmap(cmstr)
    plt.rcParams.update({'font.size': 19, 'axes.titlesize': 18, 'axes.labelsize': 18})
    i_row = 0
    # plot st vs reconstructed st, kld, hidden sates and each latent variable ------------------------------------------
    fig, ax = plt.subplots(nrows=N_ROWS, ncols=2, figsize=(25, N_ROWS * 4 + 10),
                           gridspec_kw={"width_ratios": [80, 1]})

    # plot true fhr
    ax[i_row, 1].set_axis_off()
    ax[i_row, 0].plot(t_in, signal_1, linewidth=1.5)
    ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
    ax[i_row, 0].set_xticklabels([])
    ax[i_row, 0].set_ylabel('FHR (bpm)')
    if two_channel:
        ax2 = ax[i_row, 0].twinx()
        ax2.plot(t_in, signal_2, linewidth=1.5, color="#c96b00")
        ax2.set_ylabel('UP')

    # plot latent z difference
    i_row += 1
    z_diff = np.diff(z_latent_mean, axis=1)
    z_diff_squared_sum = np.square(z_diff).sum(axis=0)
    # z_diff_sum = z_diff.sum(axis=0)
    ax[i_row, 1].set_axis_off()
    ax[i_row, 0].plot(z_diff_squared_sum, linewidth=1.5)
    ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
    ax[i_row, 0].set_xticklabels([])
    ax[i_row, 0].set_ylabel('Latent Z difference')

    i_row += 1
    imgplot = ax[i_row, 0].imshow(Sx, aspect='auto', norm="symlog",
                                  extent=[0, N / Fs, Sx.shape[0], 0])
    ax[i_row, 1].set_axis_on()
    fig.colorbar(imgplot, cax=ax[i_row, 1])
    ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
    ax[i_row, 0].set_xticklabels([])
    ax[i_row, 0].set_ylabel('True ST')

    i_row += 1
    imgplot = ax[i_row, 0].imshow(kld_values, aspect='auto', norm="symlog",
                                  extent=[0, N / Fs, kld_values.shape[0], 0])
    ax[i_row, 1].set_axis_on()
    fig.colorbar(imgplot, cax=ax[i_row, 1])
    ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
    ax[i_row, 0].set_xticklabels([])
    ax[i_row, 0].set_ylabel('KLD')

    i_row += 1
    imgplot = ax[i_row, 0].imshow(z_latent_mean, aspect='auto', norm="symlog",
                                  extent=[0, N / Fs, z_latent_mean.shape[0], 0])
    ax[i_row, 1].set_axis_on()
    fig.colorbar(imgplot, cax=ax[i_row, 1])
    ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
    ax[i_row, 0].set_xticklabels([])
    ax[i_row, 0].set_ylabel('Latent Representation Mean')

    i_row += 1
    imgplot = ax[i_row, 0].imshow(z_latent_std, aspect='auto', norm="symlog",
                                  extent=[0, N / Fs, z_latent_mean.shape[0], 0])
    ax[i_row, 1].set_axis_on()
    fig.colorbar(imgplot, cax=ax[i_row, 1])
    ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
    ax[i_row, 0].set_xticklabels([])
    ax[i_row, 0].set_ylabel('Latent Representation Std')

    if h_hidden_mean is not None:
        i_row += 1
        imgplot = ax[i_row, 0].imshow(h_hidden_mean, aspect='auto', norm="symlog",
                                      extent=[0, N / Fs, h_hidden_mean.shape[0], 0])
        ax[i_row, 1].set_axis_on()
        fig.colorbar(imgplot, cax=ax[i_row, 1])
        ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
        ax[i_row, 0].set_xticklabels([])
        ax[i_row, 0].set_ylabel('Hidden States Mean')
    t_original = np.linspace(0, 1, len(signal_1))
    t_reduced = np.linspace(0, 1, Sx.shape[1])
    for i in range(z_latent_mean.shape[0]):
        i_row += 1
        ax[i_row, 0].plot(t_reduced, z_latent_mean[i, :], linewidth=2, label="Latent Representation")
        ax2 = ax[i_row, 0].twinx()
        ax2.plot(t_original, signal_1, linewidth=1, label="fhr", color='#e0371d')
        ax[i_row, 0].fill_between(t_reduced,
                                  z_latent_mean[i, :] - z_latent_std[i, :],
                                  z_latent_mean[i, :] + z_latent_std[i, :],
                                  color='blue', alpha=0.25, label='Std dev')
        ax[i_row, 0].legend()
        ax[i_row, 1].set_axis_off()
        ax[i_row, 0].set_ylabel(f'Coefficient {i}')

    cmstr = 'Blues'
    plt.set_cmap(cmstr)
    fig.delaxes(ax[1][1])
    ax[0, 1].set_axis_off()
    # plt.savefig(plot_dir + '/' + record_name + '_' + str(domain_start[i_segment]) + '_st.pdf', bbox_inches='tight',
    #             orientation='landscape')
    plt.savefig(plot_dir + '/' + 'st-true-reconstructed' + tag + '.pdf',
                bbox_inches='tight',
                orientation='landscape',
                dpi=300)
    plt.close(fig)
    # ------------------------------------------------------------------------------------------------------------------
    # plot latent dim and histogram of it
    if plot_latent:
        i_row = 0
        N_ROWS = z_latent_mean.shape[0] + 2
        # N_ROWS = 1 * z_latent_mean.shape[0] + z_latent_mean.shape[0] + 2
        fig, ax = plt.subplots(nrows=N_ROWS, ncols=2, figsize=(25, N_ROWS * 3 + 10),
                               gridspec_kw={"width_ratios": [80, 1]})
        t_original = np.linspace(0, 1, len(signal_1))
        t_reduced = np.linspace(0, 1, Sx.shape[1])

        ax[i_row, 1].set_axis_off()
        ax[i_row, 0].plot(t_original, signal_1, linewidth=1.5, color="#3D8361")
        ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
        ax[i_row, 0].set_xticklabels([])
        ax[i_row, 0].set_ylabel(f'FHR')
        if two_channel:
            ax2 = ax[i_row, 0].twinx()
            ax2.plot(t_original, signal_2, linewidth=1.5, color="#135D66")
            ax2.set_ylabel('UP')
        # for j in range(Sx.shape[0]):
        for i in range(z_latent_mean.shape[0]):
            i_row += 1
            ax[i_row, 1].set_axis_off()
            ax2 = ax[i_row, 0].twinx()
            ax2.plot(t_original, signal_1, linewidth=1, color="#0C2D57")
            marker_line, stem_lines, baseline = ax[i_row, 0].stem(t_reduced, 1*z_latent_mean[i, :], basefmt=" ")
            plt.setp(stem_lines, 'color', "#FC6736", 'linewidth', 2)
            plt.setp(marker_line, 'color', "#387ADF")
            ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
            ax[i_row, 0].set_xticklabels([])
            ax[i_row, 0].set_ylabel(f'Latent Dim {i} Coefficient')

        i_row += 1
        ax[i_row, 1].set_axis_off()
        ax[i_row, 0].plot(Sx[0, :], linewidth=1.5, color="#0C2D57")
        marker_line, stem_lines, baseline = ax[i_row, 0].stem(1*np.mean(z_latent_mean, axis=0), basefmt=" ")
        plt.setp(stem_lines, 'color', "#FC6736", 'linewidth', 2)
        ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
        ax[i_row, 0].set_xticklabels([])
        ax[i_row, 0].set_ylabel(f'Latent Dim Averaged')

        # for i in range(z_latent_mean.shape[0]):
        #     i_row += 1
        #     ax[i_row, 1].set_axis_off()
        #     ax[i_row, 0].hist(z_latent_mean[i, :], bins=40, alpha=0.6, color='blue', rwidth=0.9, edgecolor='black')
        #     ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
        #     # ax[i_row, 0].set_xticklabels([])
        #     ax[i_row, 0].set_ylabel(f'Latent Dim Histogram {i}')

        plt.savefig(plot_dir + '/' + 'latent-representation' + tag + '.pdf',
                    bbox_inches='tight',
                    orientation='landscape',
                    dpi=50)
        plt.close(fig)
    # ------------------------------------------------------------------------------------------------------------------

    # plot hidden dims -------------------------------------------------------------------------------------------------
    if plot_state:
        i_row = 0
        # N_ROWS = 1 * h_hidden_mean.shape[0] + h_hidden_mean.shape[0] * Sx.shape[0] + 2
        N_ROWS = 1 * h_hidden_mean.shape[0] + h_hidden_mean.shape[0] * 1 + 2
        fig, ax = plt.subplots(nrows=N_ROWS, ncols=2, figsize=(25, N_ROWS * 3 + 10),
                               gridspec_kw={"width_ratios": [80, 1]})
        t_original = np.linspace(0, 1, len(signal_1))
        t_reduced = np.linspace(0, 1, Sx.shape[1])

        ax[i_row, 1].set_axis_off()
        ax[i_row, 0].plot(t_original, signal_1, linewidth=1.5, color="#3D8361")
        ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
        ax[i_row, 0].set_xticklabels([])
        ax[i_row, 0].set_ylabel(f'FHR')
        if two_channel:
            ax2 = ax[i_row, 0].twinx()
            ax2.plot(t_in, signal_2, linewidth=1.5, color="#135D66")
            ax2.set_ylabel('UP')
        # for j in range(Sx.shape[0]):
        for i in range(h_hidden_mean.shape[0]):
            i_row += 1
            ax[i_row, 1].set_axis_off()
            # ax2 = ax[i_row, 0].twinx()
            # ax2.plot(t_reduced, Sx[0, :], linewidth=1, color="#0C2D57")

            ax3 = ax[i_row, 0].twinx()
            ax3.plot(t_original, signal_1, linewidth=1.5, color="#3D8361")

            # marker_line, stem_lines, baseline = ax[i_row, 0].stem(t_reduced, 1*h_hidden_mean[i, :], basefmt=" ")
            # plt.setp(stem_lines, 'linewidth', 0)
            # plt.setp(marker_line, 'color', "#FC6736", 'marker', 'o', 'markersize', 8)
            ax[i_row, 0].plot(t_reduced, 1*h_hidden_mean[i, :], linewidth=1.5, color="#FC6736", marker='o',
                              linestyle='-', markersize=8)

            ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
            ax[i_row, 0].set_xticklabels([])
            ax[i_row, 0].set_ylabel(f'Latent Dim {i} Coefficient 0')

        i_row += 1
        ax[i_row, 1].set_axis_off()

        marker_line, stem_lines, baseline = ax[i_row, 0].stem(t_reduced, 1*np.mean(h_hidden_mean, axis=0), basefmt=" ")
        plt.setp(stem_lines, 'color', "#FC6736", 'linewidth', 0)
        plt.setp(marker_line, 'color', "#FC6736", 'marker', 'o', 'markersize', 8)
        ax2 = ax[i_row, 0].twinx()
        ax2.plot(t_reduced, Sx[0, :], linewidth=1.5, color="#0C2D57")
        ax3 = ax[i_row, 0].twinx()
        ax3.plot(t_original, signal_1, linewidth=1.5, color="#3D8361")
        ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
        ax[i_row, 0].set_xticklabels([])
        ax[i_row, 0].set_ylabel(f'Hidden Dim Averaged')

        for i in range(h_hidden_mean.shape[0]):
            i_row += 1
            ax[i_row, 1].set_axis_off()
            ax[i_row, 0].hist(h_hidden_mean[i, :], bins=40, alpha=0.6, color='blue')
            ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
            # ax[i_row, 0].set_xticklabels([])
            ax[i_row, 0].set_ylabel(f'Hidden Dim Histogram {i}')

        plt.savefig(plot_dir + '/' + tag + '_hidden' + '.pdf', bbox_inches='tight', orientation='landscape', dpi=50)
        plt.close(fig)
    # ------------------------------------------------------------------------------------------------------------------

    # plot kld values separately
    if plot_klds:
        N_ROWS = kld_values.shape[0] + 1
        fig, ax = plt.subplots(nrows=N_ROWS, ncols=2, figsize=(25, N_ROWS * 5 + 10),
                               gridspec_kw={"width_ratios": [60, 1]})
        t_1 = np.linspace(0, 10, kld_values.shape[1])
        t_2 = np.linspace(0, 10, len(signal_1))
        i_row = -1
    # for j in range(Sx.shape[0]):
        for i in range(kld_values.shape[0]):
            i_row += 1
            ax[i_row, 1].set_axis_off()
            kld_values[i, 0:10] = kld_values[i, 10]
            ax[i_row, 0].plot(t_1, kld_values[i, :], linewidth=1.5, color="#0C2D57")

            ax2 = ax[i_row, 0].twinx()
            ax3 = ax[i_row, 0].twinx()
            ax2.plot(t_2, signal_1, linewidth=1.5, color="#FE7A36")
            # ax3.plot(t_2, signal, linewidth=2, color="#0D9276")
            ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
            ax[i_row, 0].set_ylabel(f'KLD-Latent{i}')
        i_row += 1
        ax[i_row, 1].set_axis_off()
        ax[i_row, 0].plot(t_1, np.sum(kld_values, axis=0), linewidth=1.5, color="#0C2D57")
        ax3 = ax[i_row, 0].twinx()
        ax3.plot(t_2, signal_1, linewidth=1.5)

        plt.savefig(plot_dir + '/' + 'kld' + tag + '.pdf',
                    bbox_inches='tight',
                    orientation='landscape',
                    dpi=50)
        plt.close(fig)




    # N_ROWS = 2*Sx.shape[0]
    # fig, ax = plt.subplots(nrows=N_ROWS, ncols=2, figsize=(25, N_ROWS * 5 + 10),
    #                        gridspec_kw={"width_ratios": [60, 1]}, squeeze=False)
    # i_row = -1
    # for i in range(Sx.shape[0]):
    #     i_row += 1
    #     ax[i_row, 1].set_axis_off()
    #     ax[i_row, 0].plot(Sx[i, :], linewidth=2, color="#003865")
    #     ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
    #     ax[i_row, 0].set_xticklabels([])
    #     ax[i_row, 0].set_ylabel('Original Signal')
    #
    #     i_row += 1
    #     ax[i_row, 1].set_axis_off()
    #     ax[i_row, 0].plot(new_sample[i, :], linewidth=1.5, color="#EF5B0C")
    #
    #     ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
    #     ax[i_row, 0].set_xticklabels([])
    #     ax[i_row, 0].set_ylabel('New Sample')
    # plt.savefig(plot_dir + '/' + tag + '_new-sample' + '.pdf', bbox_inches='tight', orientation='landscape', dpi=100)
    # plt.close(fig)
    #
    # N_ROWS = 2 * Sx.shape[0]


def plot_latent_interpolation(len_signal, z_latent, decoder_output, plot_dir, tag):
    n_rows = len(z_latent)
    Fs = 4
    log_eps = 1e-3
    N = len_signal

    plt.set_cmap('seismic')
    plt.rcParams.update({'font.size': 19, 'axes.titlesize': 18, 'axes.labelsize': 18})

    fig, ax = plt.subplots(nrows=n_rows, ncols=2, figsize=(25, n_rows * 5 + 10), gridspec_kw={"width_ratios": [80, 1]})
    i_row = -1
    for z_c in range(z_latent.shape[0]):
        i_row += 1
        z = z_latent[z_c, :, :]
        imgplot = ax[i_row, 0].imshow(z, aspect='auto', norm="linear",
                                      extent=[0, N / Fs, z.shape[0], 0])
        ax[i_row, 1].set_axis_on()
        fig.colorbar(imgplot, cax=ax[i_row, 1])
        ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
        ax[i_row, 0].set_xticklabels([])
        ax[i_row, 0].set_ylabel('z')
    plt.savefig(plot_dir + '/' + tag + '_z_latent' + '.pdf', bbox_inches='tight', orientation='landscape', dpi=100)
    plt.close(fig)

    fig, ax = plt.subplots(nrows=n_rows, ncols=2, figsize=(25, n_rows * 5 + 10), gridspec_kw={"width_ratios": [80, 1]})
    i_row = -1
    for y_c in range(decoder_output.shape[0]):
        i_row += 1
        y = decoder_output[y_c, : , :]
        imgplot = ax[i_row, 0].imshow(y, aspect='auto', norm="linear",
                                      extent=[0, N / Fs, y.shape[0], 0])
        ax[i_row, 1].set_axis_on()
        fig.colorbar(imgplot, cax=ax[i_row, 1])
        ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
        ax[i_row, 0].set_xticklabels([])
        ax[i_row, 0].set_ylabel('y')

    plt.savefig(plot_dir + '/' + tag + '_decoder' + '.pdf', bbox_inches='tight', orientation='landscape', dpi=100)
    plt.close(fig)


def animate_latent_interpolation(len_signal, z_latent, decoder_output, plot_dir, tag):
    n_frames = z_latent.shape[0]
    Fs = 4
    N = len_signal

    plt.set_cmap('Blues')
    plt.rcParams.update({'font.size': 19, 'axes.titlesize': 18, 'axes.labelsize': 18})
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=2, figsize=(25, 7), gridspec_kw={"width_ratios": [80, 1]})

    # Initialize the plots
    img1 = ax1[0].imshow(z_latent[0], aspect='auto', norm='symlog', extent=[0, N / Fs, z_latent[0].shape[0], 0])
    cbar1 = fig.colorbar(img1, cax=ax1[1], orientation='vertical')
    ax1[0].set_ylabel('z')
    ax1[0].set_xticks([])

    img2 = ax2[0].imshow(decoder_output[0], aspect='auto', norm='symlog',
                         extent=[0, N / Fs, decoder_output[0].shape[0], 0])
    cbar2 = fig.colorbar(img2, cax=ax2[1], orientation='vertical')
    ax2[0].set_ylabel('y')
    ax2[0].set_xticks([])

    def init():
        # Initialize the images with the first frame
        img1.set_data(z_latent[0])
        img2.set_data(decoder_output[0])
        return img1, img2

    def animate(i):
        # Update the images for the ith frame
        img1.set_data(z_latent[i])
        img2.set_data(decoder_output[i])
        return img1, img2

    ani = animation.FuncAnimation(fig, animate, frames=n_frames, init_func=init, blit=True, repeat=False, interval=150)
    ani.save(f'{plot_dir}/{tag}_latent_interpolation.gif', writer='pillow', dpi=100)

    plt.close(fig)


def plot_general_mse(signal=None, plot_order=None, Sx=None, meta=None,
                     Sxr=None, Sxr_std=None, z_latent=None, plot_dir=None, tag='',
                     all_mse=None):

    cmstr = 'Blues'
    plt.set_cmap(cmstr)
    plt.rcParams.update({'font.size': 19, 'axes.titlesize': 18, 'axes.labelsize': 18})
    i_row = -1
    N_ROWS = (all_mse.shape[0])
    max_mse_value = np.max(all_mse)
    fig, ax = plt.subplots(nrows=N_ROWS, ncols=2, figsize=(25, N_ROWS * 5 + 10),
                           gridspec_kw={"width_ratios": [80, 1]})
    for i in range(N_ROWS):
        i_row += 1
        ax[i_row, 1].set_axis_off()
        ax[i_row, 0].plot(all_mse[i, :], linewidth=1.5, marker='.', ms=1)
        ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
        ax[i_row, 0].set_ylim(0, max_mse_value)
        ax[i_row, 0].set_xticklabels([])
        ax[i_row, 0].set_ylabel(f'mse coefficient {i}')
        avg_mse = np.mean(all_mse[i, :])
        ax[i_row, 0].set_title(f'Average MSE: {avg_mse:.5f}')
    plt.savefig(plot_dir + '/' + tag + '_mses' + '.pdf', bbox_inches='tight', orientation='landscape', dpi=50)
    plt.close(fig)


def plot_generated_samples(sx, sx_mean, sx_std, input_len, tag='_', plot_dir=None):
    Fs = 4
    log_eps = 1e-3
    N = input_len
    N_ROWS = 3 + (sx.shape[0])
    cmstr = 'Blues'
    plt.set_cmap(cmstr)
    plt.rcParams.update({'font.size': 19, 'axes.titlesize': 18, 'axes.labelsize': 18})
    i_row = 0

    fig, ax = plt.subplots(nrows=N_ROWS, ncols=2, figsize=(25, N_ROWS * 5 + 10),
                           gridspec_kw={"width_ratios": [80, 1]})
    imgplot = ax[i_row, 0].imshow(sx, aspect='auto', norm="symlog",
                                  extent=[0, N / Fs, sx.shape[0], 0])
    ax[i_row, 1].set_axis_on()
    fig.colorbar(imgplot, cax=ax[i_row, 1])
    ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
    ax[i_row, 0].set_xticklabels([])
    ax[i_row, 0].set_ylabel('Sample')

    i_row += 1
    imgplot = ax[i_row, 0].imshow(sx_mean, aspect='auto', norm="symlog",
                                  extent=[0, N / Fs, sx.shape[0], 0])
    ax[i_row, 1].set_axis_on()
    fig.colorbar(imgplot, cax=ax[i_row, 1])
    ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
    ax[i_row, 0].set_xticklabels([])
    ax[i_row, 0].set_ylabel('Mean')

    i_row += 1
    imgplot = ax[i_row, 0].imshow(sx_std, aspect='auto', norm="symlog",
                                  extent=[0, N / Fs, sx.shape[0], 0])
    ax[i_row, 1].set_axis_on()
    fig.colorbar(imgplot, cax=ax[i_row, 1])
    ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
    ax[i_row, 0].set_xticklabels([])
    ax[i_row, 0].set_ylabel('Standard Deviation')

    for i in range(sx.shape[0]):
        i_row += 1
        ax[i_row, 0].plot(sx_mean[i, :], linewidth=1.25, label="True")
        if sx_std is not None:
            ax[i_row, 0].fill_between(np.arange(len(sx_std[i, :])),
                                      sx_mean[i, :] - sx_std[i, :],
                                      sx_mean[i, :] + sx_std[i, :],
                                      color='blue', alpha=0.1, label='Std dev')
        ax[i_row, 0].legend()
        ax[i_row, 1].set_axis_off()
        ax[i_row, 0].set_ylabel(f'Mean and Std of Coefficient {i}')

    cmstr = 'bwr'
    plt.set_cmap(cmstr)
    fig.delaxes(ax[1][1])
    ax[0, 1].set_axis_off()
    # plt.savefig(plot_dir + '/' + record_name + '_' + str(domain_start[i_segment]) + '_st.pdf', bbox_inches='tight',
    #             orientation='landscape')
    plt.savefig(plot_dir + '/' + tag + '_' + '.pdf', bbox_inches='tight', orientation='landscape', dpi=50)
    plt.close(fig)


def plot_distributions(sx_mean=None, sx_std=None, plot_second_channel=False, plot_sample=False, plot_dir=None,
                       plot_dataset_average=False, sample_sx=None, sample_sx_mean=None, sample_sx_std=None, tag=''):
    Fs = 4
    log_eps = 1e-3
    # N = sx_mean.shape[0]
    # if Sxr is not None:
    N_ROWS = sx_mean.shape[0]
    #     # N_ROWS = 3
    #     N_ROWS = len(plot_order) + 4
    # else:
    #     # N_ROWS = 2
    #     N_ROWS = len(plot_order) + 1
    # if plot_second_channel:
    #     N_ROWS = 5 + (sx.shape[0])
    #     signal_1 = signal[:, 0]
    #     signal_2 = signal[:, 1]
    # else:
    #     N_ROWS = 4 + (Sx.shape[0])
    #     signal_2 = signal
    #     signal_1 = signal
    # t_in = np.arange(0, N) / Fs
    cmstr = 'Blues'
    plt.set_cmap(cmstr)
    plt.rcParams.update({'font.size': 7, 'axes.titlesize': 7, 'axes.labelsize': 7})

    if plot_dataset_average:
        i_row = -1
        fig, ax = plt.subplots(nrows=N_ROWS, ncols=2, figsize=(25, N_ROWS * 5 + 10),
                               gridspec_kw={"width_ratios": [80, 1]})
        for i in range(sx_mean.shape[0]):
            i_row += 1
            ax[i_row, 0].plot(sx_mean[i, :], linewidth=1.5, label=f"st_{i}")
            ax[i_row, 0].fill_between(np.arange(len(sx_mean[i, :])),
                                      sx_mean[i, :] - sx_std[i, :],
                                      sx_mean[i, :] + sx_std[i, :],
                                      color='blue', alpha=0.25, label='Std dev')
            ax[i_row, 0].legend()
            ax[i_row, 1].set_axis_off()
            ax[i_row, 0].set_ylabel(f'St-Coefficient-{i}')
        plt.savefig(plot_dir + '/' + tag + '_dataset' + '.pdf', bbox_inches='tight', orientation='landscape', dpi=50)
        plt.close(fig)

    if plot_sample:
        N_ROWS = sx_mean.shape[0]
        fig, ax = plt.subplots(nrows=N_ROWS, ncols=2, figsize=(25, N_ROWS * 5 + 10),
                               gridspec_kw={"width_ratios": [80, 1]})
        i_row = -1
        for i in range(sx_mean.shape[0]):
            i_row += 1
            ax[i_row, 0].plot(sx_mean[i, :], linewidth=1.5, linestyle=(0, (5, 1)), label=f"st_{i}")
            ax[i_row, 0].fill_between(np.arange(len(sx_mean[i, :])),
                                      sx_mean[i, :] - sx_std[i, :],
                                      sx_mean[i, :] + sx_std[i, :],
                                      color='blue', alpha=0.19, label='Std dev')
            ax[i_row, 0].plot(sample_sx_mean[i, :], linewidth=2, color='black')
            ax[i_row, 0].legend()
            ax[i_row, 1].set_axis_off()
            ax[i_row, 0].set_ylabel(f'St-Coefficient-{i}')

        plt.savefig(plot_dir + '/' + tag + '_sample' + '.pdf', bbox_inches='tight', orientation='landscape', dpi=50)
        plt.close(fig)
#
# def plot_histogram(data=None, single_channel=True, bins=100, save_dir=None, tag=''):
#     plt.rcParams.update({'font.size': 19, 'axes.titlesize': 18, 'axes.labelsize': 18})
#     if single_channel:
#         fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 14))
#         ax.hist(data, bins=bins, color='royalblue', alpha=0.99, edgecolor='black')
#         ax.plot(data, np.full_like(data, -0.01), '|', color='black', alpha=0.1, markersize=10)
#         ax.set_xlabel('Value', fontsize=14)
#         ax.set_ylabel('Frequency', fontsize=14)
#         ax.set_title('Distribution of Data with Histogram', fontsize=16)
#         ax.grid(True, linewidth=0.1)  # Reduce the grid line thickness
#         mean = np.mean(data)
#         median = np.median(data)
#         ax.axvline(mean, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean:.4f}')
#         ax.axvline(median, color='green', linestyle='dashed', linewidth=1, label=f'Median: {median:.4f}')
#         ax.legend()
#         plt.savefig(save_dir + '/' + tag + '.pdf', bbox_inches='tight',
#                     orientation='landscape', dpi=50)
#         plt.close(fig)
#     else:
#         N_ROWS = data.shape[1]
#         fig, ax = plt.subplots(nrows=N_ROWS, ncols=1, figsize=(15, N_ROWS * 1.1))
#         i_row = -1
#         for dim in range(data.shape[1]):
#             i_row += 1
#             data_i = data[:, dim]
#             ax[i_row].hist(data_i, bins=bins, color='royalblue', alpha=0.99, edgecolor='black')
#             ax[i_row].plot(data_i, np.full_like(data, -0.01), '|', color='black', alpha=0.1, markersize=10)
#             # ax[i_row].set_xlabel('Value', fontsize=14)
#             ax[i_row].set_ylabel(f'Frequency {dim}', fontsize=14)
#             # ax[i_row].set_title('Distribution of Data with Histogram', fontsize=16)
#             ax[i_row].grid(True, linewidth=0.1)  # Reduce the grid line thickness
#             mean = np.mean(data_i)
#             median = np.median(data_i)
#             ax[i_row].axvline(mean, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean:.6f}')
#             ax[i_row].axvline(median, color='green', linestyle='dashed', linewidth=1, label=f'Median: {median:.6f}')
#             ax[i_row].legend()
#         plt.savefig(save_dir + '/' + tag + '.pdf', bbox_inches='tight',
#                     orientation='landscape', dpi=50)
#         plt.close(fig)
#

# def plot_histogram(data=None, single_channel=True, bins=100, save_dir=None, tag='', num_std_dev=4):
#     plt.rcParams.update({'font.size': 7, 'axes.titlesize': 7, 'axes.labelsize': 7})
#     def filter_outliers(data, num_std_dev):
#         mean = np.mean(data)
#         std_dev = np.std(data)
#         filtered_data = data[np.abs(data - mean) <= num_std_dev * std_dev]
#         return filtered_data
#
#     if single_channel:
#         data = filter_outliers(data, num_std_dev)
#         fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 14))
#         ax.hist(data, bins=bins, color='royalblue', alpha=0.99, edgecolor='black')
#         ax.plot(data, np.full_like(data, -0.01), '|', color='black', alpha=0.1, markersize=10)
#         ax.set_xlabel('Value', fontsize=7)
#         ax.set_ylabel('Frequency', fontsize=7)
#         ax.set_title('Distribution of Data with Histogram', fontsize=7)
#         ax.grid(True, linewidth=0.1)  # Reduce the grid line thickness
#         mean = np.mean(data)
#         median = np.median(data)
#         ax.axvline(mean, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean:.4f}')
#         ax.axvline(median, color='green', linestyle='dashed', linewidth=1, label=f'Median: {median:.4f}')
#         ax.legend()
#         plt.savefig(save_dir + '/' + tag + '.png', bbox_inches='tight', orientation='landscape', dpi=1000)
#         plt.close(fig)
#     else:
#         N_ROWS = data.shape[1]
#         fig, ax = plt.subplots(nrows=N_ROWS, ncols=1, figsize=(15, N_ROWS * 1.1))
#         for dim in range(data.shape[1]):
#             data_i = filter_outliers(data[:, dim], num_std_dev)
#             ax[dim].hist(data_i, bins=bins, color='royalblue', alpha=0.99, edgecolor='black')
#             ax[dim].plot(data_i, np.full_like(data_i, -0.01), '|', color='black', alpha=0.1, markersize=10)
#             ax[dim].set_ylabel(f'Frequency {dim}', fontsize=7)
#             ax[dim].grid(True, linewidth=0.1)  # Reduce the grid line thickness
#             mean = np.mean(data_i)
#             median = np.median(data_i)
#             ax[dim].axvline(mean, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean:.6f}')
#             ax[dim].axvline(median, color='green', linestyle='dashed', linewidth=1, label=f'Median: {median:.6f}')
#             ax[dim].legend()
#         plt.savefig(save_dir + '/' + tag + '.png', bbox_inches='tight', orientation='landscape', dpi=1000)
#         plt.close(fig)

from matplotlib.lines import Line2D

def plot_histogram(data=None, single_channel=True, bins=100, save_dir=None, tag='', num_std_dev=4):
    plt.rcParams.update({
        'font.size': 7,
        'axes.titlesize': 7,
        'axes.labelsize': 7
    })

    def filter_outliers(data, num_std_dev):
        mean = np.mean(data)
        std_dev = np.std(data)
        return data[np.abs(data - mean) <= num_std_dev * std_dev]

    if single_channel:
        # Filter data for single channel
        filtered_data = filter_outliers(data, num_std_dev)

        # Create figure and axis
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 14))

        # Plot histogram
        ax.hist(filtered_data, bins=bins, color='royalblue', alpha=0.99, edgecolor='black')

        # Plot individual data points
        ax.plot(filtered_data, np.full_like(filtered_data, -0.01), '|',
                color='black', alpha=0.1, markersize=10)

        # Labels, title, grid
        ax.set_xlabel('Value', fontsize=7)
        ax.set_ylabel('Frequency', fontsize=7)
        ax.set_title('Distribution of Data with Histogram', fontsize=7)
        ax.grid(True, linewidth=0.1)

        # Compute stats
        mean     = np.mean(filtered_data)
        median   = np.median(filtered_data)
        variance = np.var(filtered_data)

        # Plot mean & median lines
        mean_line   = ax.axvline(mean,   color='red',   linestyle='dashed', linewidth=1,
                                 label=f'Mean:     {mean:.4f}')
        median_line = ax.axvline(median, color='green', linestyle='dashed', linewidth=1,
                                 label=f'Median:   {median:.4f}')

        # Dummy handle for variance
        variance_handle = Line2D([], [], linestyle='None',
                                 label=f'Variance: {variance:.4f}')

        # Legend with all three entries
        ax.legend(handles=[mean_line, median_line, variance_handle])

        # Save and close
        plt.savefig(f"{save_dir}/{tag}.png",
                    bbox_inches='tight', orientation='landscape', dpi=1000)
        plt.close(fig)

    else:
        N_ROWS = data.shape[1]
        filtered_data_list = []
        global_min = np.inf
        global_max = -np.inf

        # First pass: filter & find global bounds
        for dim in range(N_ROWS):
            data_i = filter_outliers(data[:, dim], num_std_dev)
            filtered_data_list.append(data_i)
            if data_i.size > 0:
                global_min = min(global_min, data_i.min())
                global_max = max(global_max, data_i.max())

        if global_min == np.inf or global_max == -np.inf:
            raise ValueError("All data channels are empty after filtering outliers.")

        # Create subplots
        fig, axes = plt.subplots(nrows=N_ROWS, ncols=1,
                                 figsize=(15, N_ROWS * 1.1),
                                 sharex=True)
        if N_ROWS == 1:
            axes = [axes]

        # Plot each channel
        for dim, ax in enumerate(axes):
            data_i = filtered_data_list[dim]
            ax.hist(data_i, bins=bins, color='royalblue',
                    alpha=0.99, edgecolor='black')
            ax.plot(data_i, np.full_like(data_i, -0.01), '|',
                    color='black', alpha=0.1, markersize=10)

            ax.set_xlim(global_min, global_max)
            ax.set_ylabel(f'Frequency {dim}', fontsize=7)
            ax.grid(True, linewidth=0.1)

            # Compute stats per channel
            mean_i     = np.mean(data_i)
            median_i   = np.median(data_i)
            variance_i = np.var(data_i)

            # Plot lines
            mean_line   = ax.axvline(mean_i,   color='red',   linestyle='dashed', linewidth=1,
                                     label=f'Mean:     {mean_i:.6f}')
            median_line = ax.axvline(median_i, color='green', linestyle='dashed', linewidth=1,
                                     label=f'Median:   {median_i:.6f}')

            # Dummy handle for variance
            variance_handle = Line2D([], [], linestyle='None',
                                     label=f'Variance: {variance_i:.6f}')

            ax.legend(handles=[mean_line, median_line, variance_handle])

        # Common x-label, layout, save
        plt.xlabel('Value', fontsize=7)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{tag}.png",
                    bbox_inches='tight', orientation='landscape', dpi=1000)
        plt.close(fig)


def plot_histogram_old(data=None, single_channel=True, bins=100, save_dir=None, tag='', num_std_dev=4):
    plt.rcParams.update({
        'font.size': 7,
        'axes.titlesize': 7,
        'axes.labelsize': 7
    })

    def filter_outliers(data, num_std_dev):
        mean = np.mean(data)
        std_dev = np.std(data)
        filtered_data = data[np.abs(data - mean) <= num_std_dev * std_dev]
        return filtered_data

    if single_channel:
        # Filter data for single channel
        filtered_data = filter_outliers(data, num_std_dev)

        # Create figure and axis
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 14))

        # Plot histogram
        ax.hist(filtered_data, bins=bins, color='royalblue', alpha=0.99, edgecolor='black')

        # Plot individual data points
        ax.plot(filtered_data, np.full_like(filtered_data, -0.01), '|', color='black', alpha=0.1, markersize=10)

        # Set labels and title
        ax.set_xlabel('Value', fontsize=7)
        ax.set_ylabel('Frequency', fontsize=7)
        ax.set_title('Distribution of Data with Histogram', fontsize=7)

        # Add grid
        ax.grid(True, linewidth=0.1)

        # Calculate and plot mean and median
        mean = np.mean(filtered_data)
        median = np.median(filtered_data)
        variance = np.var(filtered_data)
        ax.axvline(mean, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean:.4f}')
        ax.axvline(median, color='green', linestyle='dashed', linewidth=1, label=f'Median: {median:.4f}')
        ax.legend()

        # Save and close figure
        plt.savefig(f"{save_dir}/{tag}.png", bbox_inches='tight', orientation='landscape', dpi=1000)
        plt.close(fig)

    else:
        N_ROWS = data.shape[1]
        filtered_data_list = []
        global_min = np.inf
        global_max = -np.inf

        # First pass: Filter data and determine global min and max
        for dim in range(data.shape[1]):
            data_i = filter_outliers(data[:, dim], num_std_dev)
            filtered_data_list.append(data_i)
            if data_i.size > 0:
                current_min = data_i.min()
                current_max = data_i.max()
                if current_min < global_min:
                    global_min = current_min
                if current_max > global_max:
                    global_max = current_max

        # Handle case where all data might be empty after filtering
        if global_min == np.inf or global_max == -np.inf:
            raise ValueError("All data channels are empty after filtering outliers.")

        # Create subplots
        fig, axes = plt.subplots(nrows=N_ROWS, ncols=1, figsize=(15, N_ROWS * 1.1), sharex=True)

        # If there's only one subplot, wrap axes in a list for consistency
        if N_ROWS == 1:
            axes = [axes]

        # Plot each histogram with consistent x-axis limits
        for dim in range(N_ROWS):
            ax = axes[dim]
            data_i = filtered_data_list[dim]

            ax.hist(data_i, bins=bins, color='royalblue', alpha=0.99, edgecolor='black')
            ax.plot(data_i, np.full_like(data_i, -0.01), '|', color='black', alpha=0.1, markersize=10)
            ax.set_xlim(global_min, global_max)  # Set consistent x-axis limits
            ax.set_ylabel(f'Frequency {dim}', fontsize=7)
            ax.grid(True, linewidth=0.1)

            mean = np.mean(data_i)
            median = np.median(data_i)
            ax.axvline(mean, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean:.6f}')
            ax.axvline(median, color='green', linestyle='dashed', linewidth=1, label=f'Median: {median:.6f}')
            ax.legend()

        # Set common x-label
        plt.xlabel('Value', fontsize=7)

        # Adjust layout and save figure
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{tag}.png", bbox_inches='tight', orientation='landscape', dpi=1000)
        plt.close(fig)

def plot_forward_pass_kld(signal=None, plot_title='', Sx=None, meta=None, plot_second_channel=False,
                          Sxr=None, Sxr_std=None, z_latent=None, kld_elements=None, plot_dir=None, tag=''):
    from matplotlib.colors import Normalize
    Fs = 4
    log_eps = 1e-3
    N = len(signal)
    if plot_second_channel:
        N_ROWS = 7 + (Sx.shape[0])
        signal_1 = signal[0, :]
        signal_2 = signal[1, :]
        N = len(signal_1)
    else:
        N_ROWS = 6 + (Sx.shape[0])
        signal_2 = signal
        signal_1 = signal
        N = len(signal_1)
    t_in = np.arange(0, N) / Fs

    cmstr = 'seismic'
    plt.set_cmap(cmstr)
    plt.rcParams.update({'font.size': 19, 'axes.titlesize': 18, 'axes.labelsize': 18})

    i_row = 0

    fig, ax = plt.subplots(nrows=N_ROWS, ncols=2, figsize=(25, N_ROWS * 4 + 10),
                           gridspec_kw={"width_ratios": [80, 1]})
    ax[i_row, 1].set_axis_off()
    ax[i_row, 0].plot(t_in, signal_1, linewidth=1.5)
    ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
    ax[i_row, 0].set_xticklabels([])
    ax[i_row, 0].set_ylabel('FHR (bpm)')

    if plot_second_channel:
        i_row += 1
        ax[i_row, 1].set_axis_off()
        ax[i_row, 0].plot(t_in, signal_2, linewidth=1.5)
        ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
        ax[i_row, 0].set_xticklabels([])
        ax[i_row, 0].set_ylabel('UP')

    i_row += 1
    imgplot = ax[i_row, 0].imshow(Sx, aspect='auto', norm="symlog",
                                  extent=[0, N / Fs, Sx.shape[0], 0], interpolation='none')
    ax[i_row, 1].set_axis_on()
    fig.colorbar(imgplot, cax=ax[i_row, 1])
    ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
    ax[i_row, 0].set_xticklabels([])
    ax[i_row, 0].set_ylabel('True ST')
    from matplotlib.colors import Normalize, SymLogNorm, TwoSlopeNorm
    i_row += 1

    norm_auto = Normalize(vmin=kld_elements.min(), vmax=kld_elements.max())
    imgplot = ax[i_row, 0].imshow(kld_elements, aspect='auto', norm=norm_auto, interpolation='bilinear',
                                  extent=[0, N / Fs, Sx.shape[0], 0])
    ax[i_row, 1].set_axis_on()
    fig.colorbar(imgplot, cax=ax[i_row, 1])
    ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
    ax[i_row, 0].set_xticklabels([])
    ax[i_row, 0].set_ylabel('KLD Values')

    i_row += 1
    kld_mean = np.mean(kld_elements.T, axis=1)
    ax[i_row, 1].set_axis_off()
    ax[i_row, 0].plot(kld_mean, linewidth=3)
    ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
    ax[i_row, 0].set_xticklabels([])
    ax[i_row, 0].set_ylabel('KLD averaged')

    i_row += 1
    kld_diff = np.mean(np.diff(kld_elements.T, axis=1), axis=1)
    ax[i_row, 1].set_axis_off()
    ax[i_row, 0].plot(kld_diff, linewidth=3)
    ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
    ax[i_row, 0].set_xticklabels([])
    ax[i_row, 0].set_ylabel('KLD diffed mean')

    i_row += 1
    norm_auto = Normalize(vmin=z_latent.min(), vmax=z_latent.max())
    imgplot = ax[i_row, 0].imshow(z_latent, aspect='auto', norm=norm_auto,
                                  extent=[0, N / Fs, z_latent.shape[0], 0], interpolation='bilinear')
    ax[i_row, 1].set_axis_on()
    fig.colorbar(imgplot, cax=ax[i_row, 1])
    ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
    ax[i_row, 0].set_xticklabels([])
    ax[i_row, 0].set_ylabel('Latent Representation')

    for i in range(Sx.shape[0]):
        i_row += 1
        ax[i_row, 0].plot(Sx[i, :], linewidth=3, label="True", color="#12BA9B")
        ax[i_row, 0].plot(Sxr[i, :], linewidth=2.5, label="Reconstructed", color="#DF1619")
        if Sxr_std is not None:
            ax[i_row, 0].fill_between(np.arange(len(Sxr_std[i, :])),
                                      Sxr[i, :] - 0.7*Sxr_std[i, :],
                                      Sxr[i, :] + 0.7*Sxr_std[i, :],
                                      color='blue', alpha=0.1, label='Std dev')
        ax_right = ax[i_row, 0].twinx()
        ax_right.plot(kld_mean, linewidth=2, color="#2F97C1")
        ax[i_row, 0].autoscale(enable=True, axis='x', tight=True)
        ax_right.get_yaxis().set_visible(False)
        ax[i_row, 0].legend()
        ax[i_row, 1].set_axis_off()
        ax[i_row, 0].set_ylabel(f'Coefficient {i}')


    cmstr = 'bwr'
    plt.set_cmap(cmstr)
    fig.delaxes(ax[1][1])
    ax[0, 1].set_axis_off()
    # plt.savefig(plot_dir + '/' + record_name + '_' + str(domain_start[i_segment]) + '_st.pdf', bbox_inches='tight',
    #             orientation='landscape')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.suptitle(plot_title, fontsize=22)
    plt.savefig(plot_dir + '/st-plots' + tag + '.pdf', bbox_inches='tight', orientation='landscape', dpi=50)
    plt.close(fig)


def analyze_and_plot_classification_metrics(df_path, output_cls_test_dir):
    """
    Loads classification results from a CSV, computes overall classification metrics
    (ROC, AUROC, accuracy, precision, recall, F1, sensitivity, specificity) and then groups by
    the 'epoch_num' column (and additional flags CS and no_bg) to compute and plot these metrics
    as a function of epoch number (converted to hours).

    Parameters:
        df_path (str): Path to the CSV file containing columns:
                       - guid
                       - CS (boolean or 0/1)
                       - epoch_num (float, in seconds)
                       - prob_class_0
                       - prob_class_1
                       - raw_pred
                       - predicted_class
                       - true_label (ground truth labels, assumed to be present)
                       - no_bg (boolean or 0/1)
        output_cls_test_dir (str): Directory where all plots and metrics summaries will be saved.
    """
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    )

    # Set up plotting style
    cmstr = 'seismic'
    plt.set_cmap(cmstr)
    plt.rcParams.update({'font.size': 15, 'axes.titlesize': 11, 'axes.labelsize': 11})

    # Create output directory if it doesn't exist
    os.makedirs(output_cls_test_dir, exist_ok=True)

    # Load the results dataframe
    df = pd.read_csv(df_path)

    # --- Remove invalid epochs ---
    # Exclude the last two epoch values (-1200 and -2400)
    df = df[~df['epoch_num'].isin([-1200, -2400])]

    # Check that ground truth column exists
    if 'true_label' not in df.columns:
        raise ValueError("The input dataframe must contain a 'true_label' column with the ground truth labels.")

    # --- Overall Metrics ---
    y_true = df['true_label'].values
    y_pred = df['predicted_class'].values
    y_prob = df['prob_class_1'].values  # probability for class 1

    # Compute ROC and AUROC for overall predictions
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    overall_roc_auc = auc(fpr, tpr)

    # Plot overall ROC curve
    plt.figure(figsize=(12, 8))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {overall_roc_auc:.2f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Overall ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.ylim(0, 1)
    overall_roc_path = os.path.join(output_cls_test_dir, 'overall_roc.png')
    plt.savefig(overall_roc_path)
    plt.close()

    # Compute other overall metrics
    overall_accuracy = accuracy_score(y_true, y_pred)
    overall_precision = precision_score(y_true, y_pred, zero_division=0)
    overall_recall = recall_score(y_true, y_pred, zero_division=0)  # also sensitivity
    overall_f1 = f1_score(y_true, y_pred, zero_division=0)

    # Compute specificity using the confusion matrix: specificity = TN / (TN + FP)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    overall_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Save overall metrics to a text file
    overall_metrics = {
        'accuracy': overall_accuracy,
        'precision': overall_precision,
        'recall (sensitivity)': overall_recall,
        'specificity': overall_specificity,
        'f1_score': overall_f1,
        'roc_auc': overall_roc_auc
    }
    overall_metrics_path = os.path.join(output_cls_test_dir, 'overall_metrics.txt')
    with open(overall_metrics_path, 'w') as f:
        for metric, value in overall_metrics.items():
            f.write(f"{metric}: {value}\n")

    # --- Group Metrics by epoch_num (all data) ---
    epoch_vals_sec = []
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    aucs = []
    specificities = []

    grouped = df.groupby('epoch_num')
    for epoch, group in grouped:
        epoch_vals_sec.append(epoch)
        y_true_grp = group['true_label'].values
        y_pred_grp = group['predicted_class'].values
        y_prob_grp = group['prob_class_1'].values

        acc = accuracy_score(y_true_grp, y_pred_grp)
        prec = precision_score(y_true_grp, y_pred_grp, zero_division=0)
        rec = recall_score(y_true_grp, y_pred_grp, zero_division=0)
        f1_val = f1_score(y_true_grp, y_pred_grp, zero_division=0)

        try:
            tn_grp, fp_grp, fn_grp, tp_grp = confusion_matrix(y_true_grp, y_pred_grp).ravel()
            spec = tn_grp / (tn_grp + fp_grp) if (tn_grp + fp_grp) > 0 else np.nan
        except ValueError:
            spec = np.nan

        try:
            fpr_grp, tpr_grp, _ = roc_curve(y_true_grp, y_prob_grp)
            auc_grp = auc(fpr_grp, tpr_grp)
        except ValueError:
            auc_grp = np.nan

        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1_val)
        specificities.append(spec)
        aucs.append(auc_grp)

        # Plot ROC curve for each epoch if both classes are present
        if len(np.unique(y_true_grp)) > 1:
            plt.figure(figsize=(12, 8))
            plt.plot(fpr_grp, tpr_grp, label=f'ROC (AUC = {auc_grp:.2f})', linewidth=2)
            plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve for Epoch {epoch:.2f} sec')
            plt.legend(loc="lower right")
            plt.grid(True)
            plt.ylim(0, 1)
            epoch_roc_path = os.path.join(output_cls_test_dir, f'roc_epoch_{epoch:.2f}.png')
            plt.savefig(epoch_roc_path)
            plt.close()

    epoch_vals_hours = [sec / 3600.0 for sec in epoch_vals_sec]

    def plot_metric(metric_vals, metric_name, ylabel):
        plt.figure(figsize=(12, 7))
        plt.plot(epoch_vals_hours, metric_vals, marker='o', label=metric_name, linewidth=2)
        plt.xlabel('Epoch (hours)')
        plt.ylabel(ylabel)
        plt.title(f'{metric_name} vs Epoch (hours)')
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True)
        plt.xticks(np.arange(min(epoch_vals_hours), max(epoch_vals_hours) + 0.5, 0.5), rotation=45)
        save_path = os.path.join(output_cls_test_dir, f'{metric_name.lower().replace(" ", "_")}_vs_epoch.png')
        plt.savefig(save_path)
        plt.close()

    plot_metric(accuracies, "Accuracy", "Accuracy")
    plot_metric(precisions, "Precision", "Precision")
    plot_metric(recalls, "Recall (Sensitivity)", "Recall")
    plot_metric(specificities, "Specificity", "Specificity")
    plot_metric(f1s, "F1 Score", "F1 Score")
    plot_metric(aucs, "ROC AUC", "ROC AUC")

    group_metrics = pd.DataFrame({
        'epoch_sec':            epoch_vals_sec,
        'epoch_hours':          epoch_vals_hours,
        'accuracy':             accuracies,
        'precision':            precisions,
        'recall (sensitivity)': recalls,
        'specificity':          specificities,
        'f1_score':             f1s,
        'roc_auc':              aucs
    })
    group_metrics_path = os.path.join(output_cls_test_dir, 'group_metrics_by_epoch.csv')
    group_metrics.to_csv(group_metrics_path, index=False)

    # --- Group Metrics by (epoch_num, CS) ---
    cs_epochs = []
    cs_flags = []
    cs_accuracies = []
    cs_precisions = []
    cs_recalls = []
    cs_f1s = []
    cs_aucs = []
    cs_specificities = []

    for (epoch, cs_flag), group in df.groupby(['epoch_num', 'CS']):
        y_true_grp = group['true_label']
        y_pred_grp = group['predicted_class']
        y_prob_grp = group['prob_class_1']

        acc = accuracy_score(y_true_grp, y_pred_grp)
        prec = precision_score(y_true_grp, y_pred_grp, zero_division=0)
        rec = recall_score(y_true_grp, y_pred_grp, zero_division=0)
        f1_val = f1_score(y_true_grp, y_pred_grp, zero_division=0)
        try:
            tn_grp, fp_grp, fn_grp, tp_grp = confusion_matrix(y_true_grp, y_pred_grp).ravel()
            spec = tn_grp / (tn_grp + fp_grp) if (tn_grp + fp_grp) > 0 else np.nan
        except ValueError:
            spec = np.nan

        try:
            fpr_cs, tpr_cs, _ = roc_curve(y_true_grp, y_prob_grp)
            auc_cs = auc(fpr_cs, tpr_cs)
        except ValueError:
            auc_cs = np.nan

        cs_epochs.append(epoch)
        cs_flags.append(cs_flag)
        cs_accuracies.append(acc)
        cs_precisions.append(prec)
        cs_recalls.append(rec)
        cs_f1s.append(f1_val)
        cs_aucs.append(auc_cs)
        cs_specificities.append(spec)

    group_metrics_cs = pd.DataFrame({
        'epoch_sec':            cs_epochs,
        'CS':                   cs_flags,
        'accuracy':             cs_accuracies,
        'precision':            cs_precisions,
        'recall (sensitivity)': cs_recalls,
        'specificity':          cs_specificities,
        'f1_score':             cs_f1s,
        'roc_auc':              cs_aucs
    })
    group_metrics_cs['epoch_hours'] = group_metrics_cs['epoch_sec'] / 3600.0
    group_metrics_cs_path = os.path.join(output_cls_test_dir, 'group_metrics_by_epoch_cs.csv')
    group_metrics_cs.to_csv(group_metrics_cs_path, index=False)

    # Plot sensitivity vs epoch by CS flag
    pivot_cs_sens = group_metrics_cs.pivot(
        index='epoch_hours',
        columns='CS',
        values='recall (sensitivity)'
    )
    plt.figure(figsize=(10, 6))
    plt.plot(pivot_cs_sens.index, pivot_cs_sens[True],  marker='o', label='Sensitivity (CS=True)')
    plt.plot(pivot_cs_sens.index, pivot_cs_sens[False], marker='o', label='Sensitivity (CS=False)')
    plt.xlabel('Epoch (hours)')
    plt.ylabel('Sensitivity')
    plt.title('Sensitivity vs Epoch by CS flag')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_cls_test_dir, 'sensitivity_vs_epoch_by_CS.png'))
    plt.close()

    # Plot specificity vs epoch by CS flag
    pivot_cs_spec = group_metrics_cs.pivot(
        index='epoch_hours',
        columns='CS',
        values='specificity'
    )
    plt.figure(figsize=(10, 6))
    plt.plot(pivot_cs_spec.index, pivot_cs_spec[True],  marker='o', label='Specificity (CS=True)')
    plt.plot(pivot_cs_spec.index, pivot_cs_spec[False], marker='o', label='Specificity (CS=False)')
    plt.xlabel('Epoch (hours)')
    plt.ylabel('Specificity')
    plt.title('Specificity vs Epoch by CS flag')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_cls_test_dir, 'specificity_vs_epoch_by_CS.png'))
    plt.close()

    # --- Group Metrics by (epoch_num, no_bg) for all no_bg values ---
    nb_epochs = []
    nb_flags = []
    nb_accuracies = []
    nb_precisions = []
    nb_recalls = []
    nb_f1s = []
    nb_aucs = []
    nb_specificities = []

    for (epoch, no_bg_flag), grp in df.groupby(['epoch_num', 'no_bg']):
        y_true_nb = grp['true_label']
        y_pred_nb = grp['predicted_class']
        y_prob_nb = grp['prob_class_1']

        acc = accuracy_score(y_true_nb, y_pred_nb)
        prec = precision_score(y_true_nb, y_pred_nb, zero_division=0)
        rec = recall_score(y_true_nb, y_pred_nb, zero_division=0)
        f1_val = f1_score(y_true_nb, y_pred_nb, zero_division=0)
        try:
            tn_nb, fp_nb, fn_nb, tp_nb = confusion_matrix(y_true_nb, y_pred_nb).ravel()
            spec = tn_nb / (tn_nb + fp_nb) if (tn_nb + fp_nb) > 0 else np.nan
        except ValueError:
            spec = np.nan

        try:
            fpr_nb, tpr_nb, _ = roc_curve(y_true_nb, y_prob_nb)
            auc_nb = auc(fpr_nb, tpr_nb)
        except ValueError:
            auc_nb = np.nan

        nb_epochs.append(epoch)
        nb_flags.append(no_bg_flag)
        nb_accuracies.append(acc)
        nb_precisions.append(prec)
        nb_recalls.append(rec)
        nb_f1s.append(f1_val)
        nb_aucs.append(auc_nb)
        nb_specificities.append(spec)

    group_metrics_nb = pd.DataFrame({
        'epoch_sec':            nb_epochs,
        'no_bg':                nb_flags,
        'accuracy':             nb_accuracies,
        'precision':            nb_precisions,
        'recall (sensitivity)': nb_recalls,
        'specificity':          nb_specificities,
        'f1_score':             nb_f1s,
        'roc_auc':              nb_aucs
    })
    group_metrics_nb['epoch_hours'] = group_metrics_nb['epoch_sec'] / 3600.0

    group_metrics_nb.to_csv(
        os.path.join(output_cls_test_dir, 'group_metrics_by_epoch_nobg.csv'),
        index=False
    )

    # Plot specificity vs epoch by no_bg (both True/False)
    pivot_nb = group_metrics_nb.pivot(
        index='epoch_hours',
        columns='no_bg',
        values='specificity'
    )
    plt.figure(figsize=(10, 6))
    plt.plot(pivot_nb.index, pivot_nb[True],  marker='o', label='Specificity (no_bg=True)')
    plt.plot(pivot_nb.index, pivot_nb[False], marker='o', label='Specificity (no_bg=False)')
    plt.xlabel('Epoch (hours)')
    plt.ylabel('Specificity')
    plt.title('Specificity vs Epoch by no_bg flag')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_cls_test_dir, 'specificity_vs_epoch_by_no_bg.png'))
    plt.close()

    # --- Within no_bg == True, further split by CS ---
    nb_true_epochs = []
    nb_true_cs_flags = []
    nb_true_accuracies = []
    nb_true_precisions = []
    nb_true_recalls = []
    nb_true_f1s = []
    nb_true_aucs = []
    nb_true_specificities = []

    df_no_bg_true = df[df['no_bg'] == True]

    for (epoch, cs_flag), grp in df_no_bg_true.groupby(['epoch_num', 'CS']):
        y_true_grp = grp['true_label']
        y_pred_grp = grp['predicted_class']
        y_prob_grp = grp['prob_class_1']

        acc = accuracy_score(y_true_grp, y_pred_grp)
        prec = precision_score(y_true_grp, y_pred_grp, zero_division=0)
        rec = recall_score(y_true_grp, y_pred_grp, zero_division=0)
        f1_val = f1_score(y_true_grp, y_pred_grp, zero_division=0)
        try:
            tn_grp, fp_grp, fn_grp, tp_grp = confusion_matrix(y_true_grp, y_pred_grp).ravel()
            spec = tn_grp / (tn_grp + fp_grp) if (tn_grp + fp_grp) > 0 else np.nan
        except ValueError:
            spec = np.nan

        try:
            fpr_nbg_cs, tpr_nbg_cs, _ = roc_curve(y_true_grp, y_prob_grp)
            auc_val = auc(fpr_nbg_cs, tpr_nbg_cs)
        except ValueError:
            auc_val = np.nan

        nb_true_epochs.append(epoch)
        nb_true_cs_flags.append(cs_flag)
        nb_true_accuracies.append(acc)
        nb_true_precisions.append(prec)
        nb_true_recalls.append(rec)
        nb_true_f1s.append(f1_val)
        nb_true_aucs.append(auc_val)
        nb_true_specificities.append(spec)

    group_metrics_nobg_cs = pd.DataFrame({
        'epoch_sec':            nb_true_epochs,
        'CS':                   nb_true_cs_flags,
        'accuracy':             nb_true_accuracies,
        'precision':            nb_true_precisions,
        'recall (sensitivity)': nb_true_recalls,
        'specificity':          nb_true_specificities,
        'f1_score':             nb_true_f1s,
        'roc_auc':              nb_true_aucs
    })
    group_metrics_nobg_cs['epoch_hours'] = group_metrics_nobg_cs['epoch_sec'] / 3600.0

    group_metrics_nobg_cs.to_csv(
        os.path.join(output_cls_test_dir, 'group_metrics_by_epoch_nobg_cs.csv'),
        index=False
    )

    # Plot sensitivity vs epoch for no_bg=True, split by CS
    pivot_nobg_cs_sens = group_metrics_nobg_cs.pivot(
        index='epoch_hours',
        columns='CS',
        values='recall (sensitivity)'
    )
    plt.figure(figsize=(10, 6))
    plt.plot(pivot_nobg_cs_sens.index, pivot_nobg_cs_sens[True],  marker='o', label='Sensitivity (CS=True)')
    plt.plot(pivot_nobg_cs_sens.index, pivot_nobg_cs_sens[False], marker='o', label='Sensitivity (CS=False)')
    plt.xlabel('Epoch (hours)')
    plt.ylabel('Sensitivity')
    plt.title('Sensitivity vs Epoch for no_bg=True (split by CS)')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_cls_test_dir, 'sensitivity_vs_epoch_nobg_cs.png'))
    plt.close()

    # Plot specificity vs epoch for no_bg=True, split by CS
    pivot_nobg_cs_spec = group_metrics_nobg_cs.pivot(
        index='epoch_hours',
        columns='CS',
        values='specificity'
    )
    plt.figure(figsize=(10, 6))
    plt.plot(pivot_nobg_cs_spec.index, pivot_nobg_cs_spec[True],  marker='o', label='Specificity (CS=True)')
    plt.plot(pivot_nobg_cs_spec.index, pivot_nobg_cs_spec[False], marker='o', label='Specificity (CS=False)')
    plt.xlabel('Epoch (hours)')
    plt.ylabel('Specificity')
    plt.title('Specificity vs Epoch for no_bg=True (split by CS)')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_cls_test_dir, 'specificity_vs_epoch_nobg_cs.png'))
    plt.close()

#
# def analyze_and_plot_classification_metrics(df_path, output_cls_test_dir):
#     """
#     Loads classification results from a CSV, computes overall classification metrics
#     (ROC, AUROC, accuracy, precision, recall, F1, sensitivity, specificity) and then groups by
#     the 'epoch_num' column to compute and plot these metrics as a function of epoch number (converted to hours).
#
#     Parameters:
#         df_path (str): Path to the CSV file containing columns:
#                        - guid
#                        - epoch_num (float, in seconds)
#                        - prob_class_0
#                        - prob_class_1
#                        - predicted_class
#                        - true_label (ground truth labels, assumed to be present)
#         output_cls_test_dir (str): Directory where all plots and metrics summaries will be saved.
#     """
#     import os
#     import pandas as pd
#     import numpy as np
#     import matplotlib.pyplot as plt
#     from sklearn.metrics import (
#         roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
#     )
#     cmstr = 'seismic'
#     plt.set_cmap(cmstr)
#     plt.rcParams.update({'font.size': 15, 'axes.titlesize': 11, 'axes.labelsize': 11})
#
#     # Create output directory if it doesn't exist
#     os.makedirs(output_cls_test_dir, exist_ok=True)
#
#     # Load the results dataframe
#     df = pd.read_csv(df_path)
#
#     # --- Remove invalid epochs ---
#     # Exclude the last two epoch values (-1200 and -2400)
#     df = df[~df['epoch_num'].isin([-1200, -2400])]
#
#     # Check that ground truth column exists
#     if 'true_label' not in df.columns:
#         raise ValueError("The input dataframe must contain a 'true_label' column with the ground truth labels.")
#
#     # --- Overall Metrics ---
#     y_true = df['true_label'].values
#     y_pred = df['predicted_class'].values
#     y_prob = df['prob_class_1'].values  # probability for class 1
#
#     # Compute ROC and AUROC for overall predictions
#     fpr, tpr, _ = roc_curve(y_true, y_prob)
#     overall_roc_auc = auc(fpr, tpr)
#
#     # Plot overall ROC curve
#     plt.figure(figsize=(12, 8))
#     plt.plot(fpr, tpr, label=f'ROC curve (AUC = {overall_roc_auc:.2f})', linewidth=2)
#     plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Overall ROC Curve')
#     plt.legend(loc="lower right")
#     plt.grid(True)
#     plt.ylim(0, 1)
#     overall_roc_path = os.path.join(output_cls_test_dir, 'overall_roc.png')
#     plt.savefig(overall_roc_path)
#     plt.close()
#
#     # Compute other overall metrics
#     overall_accuracy = accuracy_score(y_true, y_pred)
#     overall_precision = precision_score(y_true, y_pred, zero_division=0)
#     overall_recall = recall_score(y_true, y_pred, zero_division=0)  # also sensitivity
#     overall_f1 = f1_score(y_true, y_pred, zero_division=0)
#
#     # Compute specificity using the confusion matrix: specificity = TN / (TN + FP)
#     tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
#     overall_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
#
#     # Save overall metrics to a text file
#     overall_metrics = {
#         'accuracy': overall_accuracy,
#         'precision': overall_precision,
#         'recall (sensitivity)': overall_recall,
#         'specificity': overall_specificity,
#         'f1_score': overall_f1,
#         'roc_auc': overall_roc_auc
#     }
#     overall_metrics_path = os.path.join(output_cls_test_dir, 'overall_metrics.txt')
#     with open(overall_metrics_path, 'w') as f:
#         for metric, value in overall_metrics.items():
#             f.write(f"{metric}: {value}\n")
#
#     # --- Group Metrics by epoch_num ---
#     # Lists to hold metric values for each epoch group
#     epoch_vals_sec = []
#     accuracies = []
#     precisions = []
#     recalls = []
#     f1s = []
#     aucs = []
#     specificities = []
#
#     # Group by epoch_num (each epoch_num value represents a signal's identifier group)
#     grouped = df.groupby('epoch_num')
#     for epoch, group in grouped:
#         epoch_vals_sec.append(epoch)
#         y_true_grp = group['true_label'].values
#         y_pred_grp = group['predicted_class'].values
#         y_prob_grp = group['prob_class_1'].values
#
#         # Compute standard metrics for this group
#         acc = accuracy_score(y_true_grp, y_pred_grp)
#         prec = precision_score(y_true_grp, y_pred_grp, zero_division=0)
#         rec = recall_score(y_true_grp, y_pred_grp, zero_division=0)  # sensitivity
#         f1_val = f1_score(y_true_grp, y_pred_grp, zero_division=0)
#
#         # Compute specificity for this group using confusion matrix, if possible
#         try:
#             tn_grp, fp_grp, fn_grp, tp_grp = confusion_matrix(y_true_grp, y_pred_grp).ravel()
#             spec = tn_grp / (tn_grp + fp_grp) if (tn_grp + fp_grp) > 0 else np.nan
#         except ValueError:
#             spec = np.nan
#
#         # Compute ROC and AUC for this group if both classes are present
#         try:
#             fpr_grp, tpr_grp, _ = roc_curve(y_true_grp, y_prob_grp)
#             auc_grp = auc(fpr_grp, tpr_grp)
#         except ValueError:
#             auc_grp = np.nan
#
#         accuracies.append(acc)
#         precisions.append(prec)
#         recalls.append(rec)
#         f1s.append(f1_val)
#         specificities.append(spec)
#         aucs.append(auc_grp)
#
#         # Optionally, plot ROC curve for each epoch if group has both classes
#         if len(np.unique(y_true_grp)) > 1:
#             plt.figure(figsize=(12, 8))
#             plt.plot(fpr_grp, tpr_grp, label=f'ROC (AUC = {auc_grp:.2f})', linewidth=2)
#             plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
#             plt.xlabel('False Positive Rate')
#             plt.ylabel('True Positive Rate')
#             plt.title(f'ROC Curve for Epoch {epoch:.2f} sec')
#             plt.legend(loc="lower right")
#             plt.grid(True)
#             plt.ylim(0, 1)
#             epoch_roc_path = os.path.join(output_cls_test_dir, f'roc_epoch_{epoch:.2f}.png')
#             plt.savefig(epoch_roc_path)
#             plt.close()
#
#
#
#     # Convert epoch seconds to hours for plotting (1 hour = 3600 seconds)
#     epoch_vals_hours = [sec / 3600.0 for sec in epoch_vals_sec]
#
#     # Function to plot metric vs epoch (in hours) with y-axis fixed to [0, 1]
#     def plot_metric(metric_vals, metric_name, ylabel):
#         plt.figure(figsize=(12, 7))
#         plt.plot(epoch_vals_hours, metric_vals, marker='o', label=metric_name, linewidth=2)
#         plt.xlabel('Epoch (hours)')
#         plt.ylabel(ylabel)
#         plt.title(f'{metric_name} vs Epoch (hours)')
#         plt.ylim(0, 1)
#         plt.legend()
#         plt.grid(True)
#         # Use reasonable x-axis ticks (e.g., using integer hours if possible)
#         plt.xticks(np.arange(min(epoch_vals_hours), max(epoch_vals_hours)+1, 0.5), rotation=45)
#         save_path = os.path.join(output_cls_test_dir, f'{metric_name.lower().replace(" ", "_")}_vs_epoch.png')
#         plt.savefig(save_path)
#         plt.close()
#
#     # Plot overall group metrics as a function of epoch
#     plot_metric(accuracies, "Accuracy", "Accuracy")
#     plot_metric(precisions, "Precision", "Precision")
#     plot_metric(recalls, "Recall (Sensitivity)", "Recall")
#     plot_metric(specificities, "Specificity", "Specificity")
#     plot_metric(f1s, "F1 Score", "F1 Score")
#     plot_metric(aucs, "ROC AUC", "ROC AUC")
#
#     # Save group metrics to a CSV file
#     group_metrics = pd.DataFrame({
#         'epoch_sec': epoch_vals_sec,
#         'epoch_hours': epoch_vals_hours,
#         'accuracy': accuracies,
#         'precision': precisions,
#         'recall (sensitivity)': recalls,
#         'specificity': specificities,
#         'f1_score': f1s,
#         'roc_auc': aucs
#     })
#     group_metrics_path = os.path.join(output_cls_test_dir, 'group_metrics_by_epoch.csv')
#     group_metrics.to_csv(group_metrics_path, index=False)
#
#
#
#
#
#
#
#
#
#
#     ####
#
#     cs_flags = []
#     epoch_vals_sec = []
#     accuracies = []
#     precisions = []
#     recalls = []
#     f1s = []
#     aucs = []
#     specificities = []
#     # … etc.
#     # cs_df = pd.read_csv(r"/data/deid/isilon/MS_model/cs_df.csv")
#     # df = df.merge(cs_df[['guid', 'CS']], on='guid', how='left')
#     # group on both keys
#     for (epoch, cs_flag), group in df.groupby(['epoch_num', 'CS']):
#         y_true = group['true_label']
#         y_pred = group['predicted_class']
#         y_prob = group['prob_class_1']
#
#         prec = precision_score(y_true, y_pred, zero_division=0)
#         # compute your metrics as before
#         acc = accuracy_score(y_true, y_pred)
#         rec = recall_score(y_true, y_pred, zero_division=0)
#         f1_val = f1_score(y_true, y_pred, zero_division=0)
#         try:
#             tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
#             spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
#         except ValueError:
#             spec = np.nan
#
#
#         try:
#             fpr__, tpr__, _ = roc_curve(y_true, y_pred)
#             auc_ = auc(fpr__, tpr__)
#         except ValueError:
#             auc_ = np.nan
#
#         # record
#         epoch_vals_sec.append(epoch)
#         cs_flags.append(cs_flag)
#         accuracies.append(acc)
#         recalls.append(rec)
#         specificities.append(spec)
#         precisions.append(prec)
#         f1s.append(f1_val)
#         aucs.append(auc_)
#         # … other metrics
#     # convert to DataFrame (so we keep CS)
#     group_metrics = pd.DataFrame({
#         'epoch_sec': epoch_vals_sec,
#         'CS': cs_flags,
#         'accuracy': accuracies,
#         'specificity': specificities,
#         'precision': precisions,
#         'recall (sensitivity)': recalls,
#         'f1_score': f1s,
#         'roc_auc': aucs
#         # … etc.
#     })
#     # add hours
#     group_metrics['epoch_hours'] = group_metrics['epoch_sec'] / 3600
#
#     # save with CS column
#     group_metrics.to_csv(os.path.join(output_cls_test_dir, 'group_metrics_by_epoch_cs.csv'),
#                          index=False)
#
#
#     # — save no_bg group metrics to CSV —
#     nb_epochs = []
#     nb_flags  = []
#     nb_specs  = []
#     epoch_vals_sec = []
#     accuracies = []
#     precisions = []
#     recalls = []
#     f1s = []
#     aucs = []
#     specificities = []
#     for (epoch, no_bg_flag), grp in df.groupby(['epoch_num', 'no_bg']):
#
#         y_true = grp['true_label']
#         y_pred = grp['predicted_class']
#         y_prob = grp['prob_class_1']
#
#         prec = precision_score(y_true, y_pred, zero_division=0)
#         # compute your metrics as before
#         acc = accuracy_score(y_true, y_pred)
#         rec = recall_score(y_true, y_pred, zero_division=0)
#         f1_val = f1_score(y_true, y_pred, zero_division=0)
#         accuracies.append(acc)
#         precisions.append(prec)
#         recalls.append(rec)
#         f1s.append(f1_val)
#         try:
#             tn, fp, fn, tp = confusion_matrix(grp['true_label'], grp['predicted_class']).ravel()
#             spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
#         except ValueError:
#             spec = np.nan
#         try:
#             fpr_nb, tpr_nb, _ = roc_curve(grp['true_label'], grp['prob_class_1'])
#             auc_nb = auc(fpr_nb, tpr_nb)
#         except ValueError:
#             auc_nb = np.nan
#         aucs.append(auc_nb)
#         nb_epochs.append(epoch)
#         nb_flags.append(no_bg_flag)
#         nb_specs.append(spec)
#
#     group_metrics_nb = pd.DataFrame({
#         'epoch_sec':   nb_epochs,
#         'no_bg':       nb_flags,
#         'accuracy': accuracies,
#         'specificity': nb_specs,
#         'precision': precisions,
#         'recall (sensitivity)': recalls,
#         'f1_score': f1s,
#         'roc_auc': aucs
#     })
#     group_metrics_nb['epoch_hours'] = group_metrics_nb['epoch_sec'] / 3600.0
#     group_metrics_nb.to_csv(
#         os.path.join(output_cls_test_dir, 'group_metrics_by_epoch_nobg.csv'),
#         index=False
#     )
#
#     pivot = group_metrics.pivot(index='epoch_hours',
#                                 columns='CS',
#                                 values='recall (sensitivity)')
#     plt.figure(figsize=(10, 6))
#     plt.plot(pivot.index, pivot[True], marker='o', label='Sensitivity (CS=True)')
#     plt.plot(pivot.index, pivot[False], marker='o', label='Sensitivity (CS=False)')
#     plt.xlabel('Epoch (hours)')
#     plt.ylabel('Sensitivity')
#     plt.title('Sensitivity vs Epoch by CS flag')
#     plt.ylim(0, 1)
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(os.path.join(output_cls_test_dir, 'sensitivity_vs_epoch_by_CS.png'))
#     plt.close()
#
#     pivot = group_metrics.pivot(index='epoch_hours',
#                                 columns='CS',
#                                 values='specificity')
#     plt.figure(figsize=(10, 6))
#     plt.plot(pivot.index, pivot[True], marker='o', label='Specificity (CS=True)')
#     plt.plot(pivot.index, pivot[False], marker='o', label='Specificity (CS=False)')
#     plt.xlabel('Epoch (hours)')
#     plt.ylabel('Specificity')
#     plt.title('Specificity vs Epoch by CS flag')
#     plt.ylim(0, 1)
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(os.path.join(output_cls_test_dir, 'Specificity_vs_epoch_by_CS.png'))
#     plt.close()
#
#     plt.close()
#     ###
#
#     pivot_nb = group_metrics_nb.pivot(index='epoch_hours',
#                                    columns='no_bg',
#                                    values='specificity')
#     plt.figure(figsize=(10, 6))
#     plt.plot(pivot_nb.index, pivot_nb[True], marker='o', label='Specificity (no_bg=True)')
#     plt.plot(pivot_nb.index, pivot_nb[False], marker='o', label='Specificity (no_bg=False)')
#     plt.xlabel('Epoch (hours)')
#     plt.ylabel('Specificity')
#     plt.title('Specificity vs Epoch by no_bg flag')
#     plt.ylim(0, 1)
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(os.path.join(output_cls_test_dir, 'Specificity_vs_epoch_by_no_bg.png'))
#     plt.close()
#
#
#
#     # --- Group metrics by no_bg flag ---
#     # from sklearn.metrics import confusion_matrix
#     #
#     # no_bg_flags, epoch_vals_sec, specificities_nb = [], [], []
#     #
#     # for (epoch, no_bg_flag), grp in df.groupby(['epoch_num', 'no_bg']):
#     #     # compute specificity = TN / (TN + FP)
#     #     tn, fp, fn, tp = confusion_matrix(grp['true_label'], grp['predicted_class']).ravel()
#     #     spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
#     #
#     #     epoch_vals_sec.append(epoch)
#     #     no_bg_flags.append(no_bg_flag)
#     #     specificities_nb.append(spec)
#     #
#     # df_nb = pd.DataFrame({
#     #     'epoch_sec':       epoch_vals_sec,
#     #     'no_bg':           no_bg_flags,
#     #     'specificity':     specificities_nb
#     # })
#     # df_nb['epoch_hours'] = df_nb['epoch_sec'] / 3600.0
#     #
#     # # save per‐fold CSV
#     # nb_csv = os.path.join(output_cls_test_dir, 'group_metrics_by_epoch_no_bg.csv')
#     # df_nb.to_csv(nb_csv, index=False)
#     #
#     # # plot specificity vs epoch by no_bg
#     # pivot_nb = df_nb.pivot(index='epoch_hours', columns='no_bg', values='specificity')
#     # plt.figure(figsize=(10,6))
#     # plt.plot(pivot_nb.index, pivot_nb[True],  marker='o', label='Specificity (no_bg=True)')
#     # plt.plot(pivot_nb.index, pivot_nb[False], marker='o', label='Specificity (no_bg=False)')
#     # plt.xlabel('Epoch (hours)')
#     # plt.ylabel('Specificity')
#     # plt.title('Specificity vs Epoch by no_bg flag')
#     # plt.ylim(0,1)
#     # plt.legend()
#     # plt.grid(True, linestyle='--', alpha=0.3)
#     # plt.savefig(os.path.join(output_cls_test_dir, 'specificity_vs_epoch_by_no_bg.png'))
#     # plt.close()
#
#
#     print(f"Overall and per-epoch classification metrics and plots have been saved to {output_cls_test_dir}")


def analyze_class_stats_and_plot(df_path, output_cls_test_dir):
    """
    Loads classification results from a CSV file, calculates class-specific statistics, and generates plots.

    In particular:
      - Converts epoch numbers (in seconds) to minutes.
      - Assumes that epoch minutes are in intervals of -20 (e.g. -20, -40, -60, ...).
      - Computes how many positive (true_label == 1) and negative (true_label == 0) samples are present per epoch (in minutes).
      - Plots a bar chart (side-by-side) of counts per epoch minute.
      - Plots histograms of epoch (in minutes) distributions for positive and negative samples using optimized bins.
      - Saves overall class counts and percentages to a text file.

    Parameters:
        df_path (str): Path to the CSV file containing columns:
                       - guid
                       - epoch_num (float, in seconds)
                       - prob_class_0
                       - prob_class_1
                       - predicted_class
                       - true_label (ground truth labels)
        output_cls_test_dir (str): Directory where the plots and stats files will be saved.
    """
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    cmstr = 'seismic'
    plt.set_cmap(cmstr)
    plt.rcParams.update({'font.size': 15, 'axes.titlesize': 11, 'axes.labelsize': 11})

    # Create output directory if it doesn't exist
    os.makedirs(output_cls_test_dir, exist_ok=True)

    # Load the dataframe
    df = pd.read_csv(df_path)

    # Verify necessary columns exist
    required_columns = ['epoch_num', 'true_label', 'predicted_class']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' is missing from the input dataframe.")

    # Convert epoch numbers from seconds to minutes (rounded to nearest multiple of 20)
    # Since epochs are at -20 minute intervals, we round to the nearest multiple of -20.
    # First convert seconds to minutes
    df['epoch_min'] = df['epoch_num'] / 60.0
    # Then round to nearest 20-minute interval.
    # For negative numbers, np.round with a negative divisor works as expected.
    df['epoch_min'] = (np.round(df['epoch_min'] / 20) * 20).astype(int)

    # ------------------------------
    # Calculate Class Distribution per Epoch (in minutes)
    # ------------------------------
    # Group by epoch_min and true_label, and count the number of samples per group.
    group_counts = df.groupby(['epoch_min', 'true_label']).size().unstack(fill_value=0)
    # Ensure both classes appear in the columns (0 for negative, 1 for positive)
    for label in [0, 1]:
        if label not in group_counts.columns:
            group_counts[label] = 0
    group_counts = group_counts.sort_index()

    # Plot a bar chart showing counts per epoch (in minutes)
    ax = group_counts.plot(kind='bar', figsize=(12, 7))
    ax.set_xlabel("Epoch (minutes)")
    ax.set_ylabel("Count")
    ax.set_title("Sample Counts per Epoch (minutes) by True Label")
    ax.set_ylim(0, group_counts.values.max() * 1.1)
    ax.legend(title='True Label', labels=['Negative', 'Positive'])
    # Set x-axis ticks to show the negative intervals (e.g., -20, -40, -60, ...)
    ax.set_xticklabels(group_counts.index.astype(str), rotation=45)
    bar_chart_path = os.path.join(output_cls_test_dir, "class_counts_per_epoch_min.png")
    plt.tight_layout()
    plt.savefig(bar_chart_path)
    plt.close()

    # ------------------------------
    # Plot Histograms of Epoch (minutes) Distribution for Each Class
    # ------------------------------
    # Determine the range for bins. Assuming epochs are in intervals of -20 minutes,
    # we create bins from the minimum to maximum epoch_min with a step of 20.
    min_epoch = df['epoch_min'].min()
    max_epoch = df['epoch_min'].max()
    # Make sure bins cover the entire range with a step of 20 minutes.
    bins = np.arange(min_epoch, max_epoch + 20, 20)

    plt.figure(figsize=(12, 7))
    # Plot histogram for negatives (true_label == 0)
    plt.hist(df[df['true_label'] == 0]['epoch_min'], bins=bins, alpha=0.7, label='Negative')
    # Plot histogram for positives (true_label == 1)
    plt.hist(df[df['true_label'] == 1]['epoch_min'], bins=bins, alpha=0.7, label='Positive')
    plt.xlabel("Epoch (minutes)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Epoch (minutes) Distribution by True Label")
    plt.legend()
    plt.xticks(bins, rotation=45)
    plt.tight_layout()
    epoch_hist_path = os.path.join(output_cls_test_dir, "epoch_histogram_by_class.png")
    plt.savefig(epoch_hist_path)
    plt.close()

    # ------------------------------
    # Calculate Overall Class Statistics
    # ------------------------------
    total_samples = len(df)
    class_counts_overall = df['true_label'].value_counts().rename(index={0: 'Negative', 1: 'Positive'})
    class_percentages = (class_counts_overall / total_samples * 100).round(2)

    # Write overall stats to a text file
    stats_file_path = os.path.join(output_cls_test_dir, "class_stats.txt")
    with open(stats_file_path, 'w') as f:
        f.write("Overall Class Statistics\n")
        f.write(f"Total Samples: {total_samples}\n\n")
        for label in class_counts_overall.index:
            count = class_counts_overall[label]
            perc = class_percentages[label]
            f.write(f"{label}: {count} samples ({perc}%)\n")

    # (Optional) Additional Plot: Distribution of Class 1 Probabilities by True Label
    if 'prob_class_1' in df.columns:
        plt.figure(figsize=(12, 7))
        plt.hist(df[df['true_label'] == 0]['prob_class_1'], bins=30, alpha=0.7, label='Negative')
        plt.hist(df[df['true_label'] == 1]['prob_class_1'], bins=30, alpha=0.7, label='Positive')
        plt.xlabel("Probability for Class 1")
        plt.ylabel("Frequency")
        plt.title("Distribution of Class 1 Probabilities by True Label")
        plt.legend()
        plt.tight_layout()
        prob_hist_path = os.path.join(output_cls_test_dir, "class1_probability_distribution.png")
        plt.savefig(prob_hist_path)
        plt.close()

    print(f"Class statistics and plots have been saved to {output_cls_test_dir}")


