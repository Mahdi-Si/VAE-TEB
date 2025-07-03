import tensorflow as tf

from ...backend.tensorflow_backend import TensorFlowBackend


class TensorFlowBackend1D(TensorFlowBackend):
    @classmethod
    def subsample_fourier(cls, x, k):
        """Subsampling in the Fourier domain
        Subsampling in the temporal domain amounts to periodization in the Fourier
        domain, so the input is periodized according to the subsampling factor.
        Parameters
        ----------
        x : tensor
            Input tensor with at least 3 dimensions, where the next to last
            corresponds to the frequency index in the standard PyTorch FFT
            ordering. The length of this dimension should be a power of 2 to
            avoid errors. The last dimension should represent the real and
            imaginary parts of the Fourier transform.
        k : int
            The subsampling factor.
        Returns
        -------
        res : tensor
            The input tensor periodized along the next to last axis to yield a
            tensor of size x.shape[-2] // k along that dimension.
        """
        cls.complex_check(x)

        y = tf.reshape(x, (-1, k, x.shape[-1] // k))

        return tf.reduce_mean(y, axis=-2)

    @staticmethod
    def pad(x, pad_left, pad_right):
        """Pad real 1D tensors
        1D implementation of the padding function for real PyTorch tensors.
        Parameters
        ----------
        x : tensor
            Three-dimensional input tensor with the third axis being the one to
            be padded.
        pad_left : int
            Amount to add on the left of the tensor (at the beginning of the
            temporal axis).
        pad_right : int
            amount to add on the right of the tensor (at the end of the temporal
            axis).
        Returns
        -------
        res : tensor
            The tensor passed along the third dimension.
        """
        if (pad_left >= x.shape[-1]) or (pad_right >= x.shape[-1]):
            raise ValueError('Indefinite padding size (larger than tensor).')

        paddings = [[0, 0]] * len(x.shape[:-1])
        paddings += [[pad_left, pad_right]]

        return tf.pad(x, paddings, mode="REFLECT")

    @staticmethod
    def unpad(x, i0, i1):
        """Unpad real 1D tensor
        Slices the input tensor at indices between i0 and i1 along the last axis.
        Parameters
        ----------
        x : tensor
            Input tensor with least one axis.
        i0 : int
            Start of original signal before padding.
        i1 : int
            End of original signal before padding.
        Returns
        -------
        x_unpadded : tensor
            The tensor x[..., i0:i1].
        """
        return x[..., i0:i1]

    @classmethod
    def rfft(cls, x):
        #cls.real_check(x)

        #return tf.signal.fft(tf.cast(x, tf.complex64), name='rfft1d')
        return tf.signal.fft(cls.cast_complex(x), name='rfft1d')

    @classmethod
    def irfft(cls, x):
        cls.complex_check(x)

        return tf.math.real(tf.signal.ifft(x, name='irfft1d'))

    @classmethod
    def ifft(cls, x):
        cls.complex_check(x)

        return tf.signal.ifft(x, name='ifft1d')
    
    @classmethod
    def fft(cls, x):

        return tf.signal.fft(x, name='fft1d')

    @classmethod
    def cast_complex(cls, x):
        cls.real_check(x)
        
        #return tf.cast(x, tf.complex64, name='cast_complex1d')
        return tf.complex(x, tf.zeros_like(x), name='cast_complex1d')
      
    #@staticmethod
    #def cast_complex(x):
        
    #    return tf.cast(x, tf.complex64, name='cast_complex1d')

    # @staticmethod
    # @tf.function
    # def my_pow(a, b):
      # return a ** b
    
 
    @classmethod
    def pow(cls, x, power):
        cls.complex_check(x)
        
        #result = tf.math.pow(x, power, name='pow1d') # fails: power needs to be complex

        #log_x = tf.math.log(x, name='log1d') # log operator is placed on CPU (but exp is not!)
        #power_log_x = cls.multiply(power, log_x)
        #result = tf.math.exp(power_log_x, name='exp1d')        

        #power = tf.constant(power, dtype=tf.complex64) # this gets created on the CPU 
        
        power = cls.cast_complex(power*tf.ones_like(power))
        #power = tf.complex(power*tf.ones_like(power), tf.zeros_like(power))
        result = tf.math.pow(x, power, name='pow1d') 
        return result
      
    @classmethod
    def pow_polar(cls, x, power):
        cls.complex_check(x)
        
        angle = tf.math.angle(x, name='angle1d')
        power = power * tf.ones_like(power)
        power_angle = cls.multiply(angle, power)
        real = tf.math.cos(power_angle, name='cos1d')
        imag = tf.math.sin(power_angle, name='sin1d')
        result = tf.complex(real, imag, name='complex_pow_1d')
        return result
      
    @classmethod
    def conj(cls, x): 
        #result = tf.math.conj(x, name='conj1d')
        # to fix when tf ?2.2.x? bug on Windows is fixed
        # https://github.com/tensorflow/tensorflow/issues/38443
        b = tf.math.real(x, name='conj_r_1d')
        #c = tf.math.imag(x, name='conj_i_1d') * -1
        c = tf.math.imag(x, name='conj_i_1d') * tf.constant(-1, dtype=tf.float32)
        x = tf.complex(b, c, name='conj_c_1d')
        
        return x
    
    @classmethod
    def real(cls, x):

        return tf.math.real(x)

    @classmethod
    def divide(cls, x, y):

        return tf.divide(x, y, name='divide1d')

    @classmethod
    def multiply(cls, x, y):

        return tf.multiply(x, y, name='multiply1d')

    @classmethod
    def modulo(cls, x, y):

        return tf.math.floormod(x, y, name='modulo1d')
    
    
    @classmethod
    def reshape(cls, x, new_shape):

        return tf.reshape(x, new_shape)

    @classmethod
    def to_polar(cls, x):
        cls.complex_check(x)
        
        mag = cls.modulus(x)
        phase = tf.math.angle(x, name='angle1d')
        return mag, phase
      
    @classmethod
    def to_cartesian(cls, mag, phase):
        # temp!!!
        #cls.real_check(mag)
        cls.real_check(phase)

        real = tf.math.cos(phase, name='to_cartesion_cos1d')
        real = cls.multiply(real, mag)
        imag = tf.math.sin(phase, name='to_cartesion_sin1d')
        imag = cls.multiply(imag, mag)
        return tf.complex(real, imag, name='to_cartesion_c_1d')
      
    @classmethod
    def rms_diff(cls, x, y):
        rms = tf.math.sqrt(tf.reduce_mean(tf.math.squared_difference(x,y)))
        print('rms= ' + str(rms))
        #assert diff < eps
          


      





backend = TensorFlowBackend1D
