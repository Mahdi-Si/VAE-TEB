import torch
import warnings

from ...frontend.torch_frontend import ScatteringTorch
from ..core.scattering1d import scattering1d
from ..utils import precompute_size_scattering
from .base_frontend import ScatteringBase1D


class ScatteringTorch1D(ScatteringTorch, ScatteringBase1D):
    def __init__(self, J, shape, Q=1, max_order=2, average=True,
            oversampling=0, vectorize=True, out_type='array', backend='torch', T=None):
        ScatteringTorch.__init__(self)
        ScatteringBase1D.__init__(self, J, shape, Q, max_order, average,
                oversampling, T, vectorize, out_type, backend=backend)
        ScatteringBase1D._instantiate_backend(self, 'kymatio.scattering1d.backend.')
        ScatteringBase1D.build(self)
        ScatteringBase1D.create_filters(self)
        self.register_filters()
        self.T=T

    # def register_filters(self):
    #     """ This function run the filterbank function that
    #     will create the filters as numpy array, and then, it
    #     saves those arrays as module's buffers."""
    #     n = 0
    #     # prepare for pytorch
    #     for k in self.phi_f.keys():
    #         if type(k) != str:
    #             # view(-1, 1).repeat(1, 2) because real numbers!
    #             self.phi_f[k] = torch.from_numpy(
    #                 self.phi_f[k]).float().view(-1, 1)
    #             self.register_buffer('tensor' + str(n), self.phi_f[k])
    #             n += 1
    #     for psi_f in self.psi1_f:
    #         for sub_k in psi_f.keys():
    #             if type(sub_k) != str:
    #                 # view(-1, 1).repeat(1, 2) because real numbers!
    #                 psi_f[sub_k] = torch.from_numpy(
    #                     psi_f[sub_k]).float().view(-1, 1)
    #                 self.register_buffer('tensor' + str(n), psi_f[sub_k])
    #                 n += 1
    #     for psi_f in self.psi2_f:
    #         for sub_k in psi_f.keys():
    #             if type(sub_k) != str:
    #                 # view(-1, 1).repeat(1, 2) because real numbers!
    #                 psi_f[sub_k] = torch.from_numpy(
    #                     psi_f[sub_k]).float().view(-1, 1)
    #                 self.register_buffer('tensor' + str(n), psi_f[sub_k])
    #                 n += 1
    #
    # def load_filters(self):
    #     """This function loads filters from the module's buffer """
    #     buffer_dict = dict(self.named_buffers())
    #     n = 0
    #
    #     for k in self.phi_f.keys():
    #         if type(k) != str:
    #             self.phi_f[k] = buffer_dict['tensor' + str(n)]
    #             n += 1
    #
    #     for psi_f in self.psi1_f:
    #         for sub_k in psi_f.keys():
    #             if type(sub_k) != str:
    #                 psi_f[sub_k] = buffer_dict['tensor' + str(n)]
    #                 n += 1
    #
    #     for psi_f in self.psi2_f:
    #         for sub_k in psi_f.keys():
    #             if type(sub_k) != str:
    #                 psi_f[sub_k] = buffer_dict['tensor' + str(n)]
    #                 n += 1

    # Dev implementation------------------------------------------------------------------------------------------------
    def register_filters(self):
        """ This function run the filterbank function that
        will create the filters as numpy array, and then, it
        saves those arrays as module's buffers."""
        n = 0
        # prepare for pytorch
        for level in range(len(self.phi_f['levels'])):
            self.phi_f['levels'][level] = torch.from_numpy(
                self.phi_f['levels'][level]).float().view(-1, 1)
            self.register_buffer('tensor' + str(n), self.phi_f['levels'][level])
            n += 1
        for psi_f in self.psi1_f:
            for level in range(len(psi_f['levels'])):
                psi_f['levels'][level] = torch.from_numpy(
                    psi_f['levels'][level]).float().view(-1, 1)
                self.register_buffer('tensor' + str(n), psi_f['levels'][level])
                n += 1
        for psi_f in self.psi2_f:
            for level in range(len(psi_f['levels'])):
                psi_f['levels'][level] = torch.from_numpy(
                    psi_f['levels'][level]).float().view(-1, 1)
                self.register_buffer('tensor' + str(n), psi_f['levels'][level])
                n += 1

    def load_filters(self):
        """This function loads filters from the module's buffer """
        buffer_dict = dict(self.named_buffers())
        n = 0

        for level in range(len(self.phi_f['levels'])):
            self.phi_f['levels'][level] = buffer_dict['tensor' + str(n)]
            n += 1

        for psi_f in self.psi1_f:
            for level in range(len(psi_f['levels'])):
                psi_f['levels'][level] = buffer_dict['tensor' + str(n)]
                n += 1

        for psi_f in self.psi2_f:
            for level in range(len(psi_f['levels'])):
                psi_f['levels'][level] = buffer_dict['tensor' + str(n)]
                n += 1

    # new implementation by Mahdi
    # def register_filters(self):
    #     n = 0
    #     for i, real_np in enumerate(self.phi_f['levels']):
    #         real = torch.from_numpy(real_np).float().view(-1, 1)
    #         complex_filter = torch.cat([real, torch.zeros_like(real)], dim=-1)  # now shape (N,2)
    #         self.register_buffer(f"tensor{n}", complex_filter)
    #         self.phi_f['levels'][i] = complex_filter
    #         n += 1
    #
    #     for psi_f in self.psi1_f:
    #         for j, real_np in enumerate(psi_f['levels']):
    #             real = torch.from_numpy(real_np).float().view(-1, 1)
    #             cf = torch.cat([real, torch.zeros_like(real)], dim=-1)
    #             self.register_buffer(f"tensor{n}", cf)
    #             psi_f['levels'][j] = cf
    #             n += 1
    #
    #     for psi_f in self.psi2_f:
    #         for j, real_np in enumerate(psi_f['levels']):
    #             real = torch.from_numpy(real_np).float().view(-1, 1)
    #             cf = torch.cat([real, torch.zeros_like(real)], dim=-1)
    #             self.register_buffer(f"tensor{n}", cf)
    #             psi_f['levels'][j] = cf
    #             n += 1
    #
    # def load_filters(self):
    #     buffers = dict(self.named_buffers())
    #     n = 0
    #
    #     for i in range(len(self.phi_f['levels'])):
    #         self.phi_f['levels'][i] = buffers[f"tensor{n}"]
    #         n += 1
    #
    #     for psi_f in self.psi1_f:
    #         for j in range(len(psi_f['levels'])):
    #             psi_f['levels'][j] = buffers[f"tensor{n}"]
    #             n += 1
    #
    #     for psi_f in self.psi2_f:
    #         for j in range(len(psi_f['levels'])):
    #             psi_f['levels'][j] = buffers[f"tensor{n}"]
    #             n += 1

    # ------------------------------------------------------------------------------------------------------------------
    def scattering(self, x):
        # basic checking, should be improved
        if len(x.shape) < 1:
            raise ValueError(
                'Input tensor x should have at least one axis, got {}'.format(
                    len(x.shape)))

        if not self.out_type in ('array', 'list'):
            raise RuntimeError("The out_type must be one of 'array' or 'list'.")

        if not self.average and self.out_type == 'array' and self.vectorize:
            raise ValueError("Options average=False, out_type='array' and "
                             "vectorize=True are mutually incompatible. "
                             "Please set out_type to 'list' or vectorize to "
                             "False.")

        if not self.vectorize:
            warnings.warn("The vectorize option is deprecated and will be "
                          "removed in version 0.3. Please set "
                          "out_type='list' for equivalent functionality.",
                          DeprecationWarning)

        batch_shape = x.shape[:-1]
        signal_shape = x.shape[-1:]

        x = x.reshape((-1, 1) + signal_shape)

        self.load_filters()

        # get the arguments before calling the scattering
        # treat the arguments
        if self.vectorize:
            size_scattering = precompute_size_scattering(
                self.J, self.Q, max_order=self.max_order, detail=True, T=self.T)
        else:
            size_scattering = 0


        # S = scattering1d(x, self.backend.pad, self.backend.unpad, self.backend, self.J, self.psi1_f, self.psi2_f, self.phi_f,\
        #                  max_order=self.max_order, average=self.average,
        #                pad_left=self.pad_left, pad_right=self.pad_right,
        #                ind_start=self.ind_start, ind_end=self.ind_end,
        #                oversampling=self.oversampling,
        #                vectorize=self.vectorize,
        #                size_scattering=size_scattering,
        #                out_type=self.out_type)

        #######################################################
        # [S, P] = scattering1d(x, self.backend.pad, self.backend.unpad, self.backend, self.J, self.T, self.psi1_f,
        #                       self.psi2_f,
        #                       self.phi_f, max_order=self.max_order, average=self.average, pad_left=self.pad_left,
        #                       pad_right=self.pad_right, ind_start=self.ind_start, ind_end=self.ind_end,
        #                       oversampling=self.oversampling,
        #                       vectorize=self.vectorize,
        #                       size_scattering=size_scattering,
        #                       out_type=self.out_type)
        [S, P] = scattering1d(x, self.backend.pad, self.backend.unpad, self.backend, self.J, self.T, self.psi1_f,
                              self.psi2_f,
                              self.phi_f, max_order=self.max_order, average=self.average, pad_left=self.pad_left,
                              pad_right=self.pad_right, ind_start=self.ind_start, ind_end=self.ind_end,
                              oversampling=self.oversampling,
                              vectorize=self.vectorize,
                              size_scattering=size_scattering,
                              out_type=self.out_type,
                              do_phase_correlation=False)

        ########################################################

        if self.out_type == 'array' and self.vectorize:
            scattering_shape = S.shape[-2:]
            new_shape = batch_shape + scattering_shape

            S = S.reshape(new_shape)
            #
            # phase_shape = S.shape(P)[-2:]
            # new_shape = tf.concat((batch_shape, phase_shape), 0)
            # P = tf.reshape(P, new_shape)
        elif self.out_type == 'array' and not self.vectorize:
            for k, v in S.items():
                # NOTE: Have to get the shape for each one since we may have
                # average == False.
                scattering_shape = v.shape[-2:]
                new_shape = batch_shape + scattering_shape

                S[k] = v.reshape(new_shape)
        elif self.out_type == 'list':
            for x in S:
                scattering_shape = x['coef'].shape[-1:]
                new_shape = batch_shape + scattering_shape

                x['coef'] = x['coef'].reshape(new_shape)

        return [S, P]


ScatteringTorch1D._document()


__all__ = ['ScatteringTorch1D']
