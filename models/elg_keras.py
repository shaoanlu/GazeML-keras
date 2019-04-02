from keras.layers import *
from keras.models import Model
import tensorflow as tf
import numpy as np

class KerasELG():
    def __init__(self, first_layer_stride=3, hg_num_feature_maps=64, hg_num_modules=3):
        self._first_layer_stride = first_layer_stride
        self._hg_num_feature_maps = hg_num_feature_maps
        self._hg_num_modules = hg_num_modules
        self._hg_num_residual_blocks = 1
        self._hg_num_landmarks = 18
        
        self.net = self.build_elg_network()
        
    def build_elg_network(self):
        return self.elg()
    
    """
    The following code is heavily refer to GazeML source code:
    https://github.com/swook/GazeML/blob/master/src/models/elg.py
    """
    def elg(self):
        outputs = {}
        inp = Input((108, 180, 1))
        
        # Prepare for Hourglass by downscaling via conv
        n = self._hg_num_feature_maps
        pre_conv1 = self._apply_conv(inp, n, k=7, s=self._first_layer_stride, name="hourglass_pre")
        pre_conv1 = self._apply_bn(pre_conv1, name="hourglass_pre_BatchNorm")        
        pre_conv1 = Activation('relu')(pre_conv1)
        pre_res1 = self._build_residual_block(pre_conv1, 2*n, name="hourglass_pre_res1")
        pre_res2 = self._build_residual_block(pre_res1, n, name="hourglass_pre_res2")
        
        # Hourglass blocks
        x = pre_res2
        x_prev = pre_res2
        for i in range(self._hg_num_modules):
            prefix = f"hourglass_hg_{str(i+1)}"
            x = self._build_hourglass(x, steps_to_go=4, f=self._hg_num_feature_maps, name=prefix)
            x, h = self._build_hourglass_after(x_prev, x, do_merge=(i<(self._hg_num_modules-1)), name=prefix)
            x_prev = x
        x = h
        outputs['heatmaps'] = x
        
        return Model(inp, outputs['heatmaps'])
        
    def _apply_conv(self, x, f, k=3, s=1, padding='same', name=None):
        return Conv2D(f, kernel_size=k, strides=s, use_bias=True, padding=padding, name=name)(x)
    
    def _apply_bn(self, x, name=None):
        return BatchNormalization(name=name)(x)
    
    def _apply_pool(self, x, k=2, s=2):
        return MaxPooling2D(pool_size=k, strides=s, padding="same")(x)
    
    def _reflect_padding_2d(self, x, pad=1):
        x = Lambda(lambda x: tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode='REFLECT'))(x)
        return x
    
    def _build_residual_block(self, x, f, name="res_block"):
        num_in = x.shape.as_list()[-1]
        half_num_out = max(int(f/2), 1)
        c = x
        conv1 = self._apply_bn(c, name=name+"_conv1_BatchNorm")
        conv1 = Activation('relu')(conv1)
        conv1 = self._apply_conv(conv1, half_num_out, k=1, s=1, name=name+"_conv1")
        conv2 = self._apply_bn(conv1, name=name+"_conv2_BatchNorm")
        conv2 = Activation('relu')(conv2)
        conv2 = self._apply_conv(conv2, half_num_out, k=3, s=1, name=name+"_conv2")
        conv3 = self._apply_bn(conv2, name=name+"_conv3_BatchNorm")
        conv3 = Activation('relu')(conv3)
        conv3 = self._apply_conv(conv3, f, k=1, s=1, name=name+"_conv3")
        
        if num_in == f:
            s = x
        else:
            s = self._apply_conv(x, f, k=1, s=1, name=name+"_skip")       
        out = Add()([conv3, s])
        return out
    
    def _build_hourglass(self, x, steps_to_go, f, depth=1, name=None):
        prefix_name = name + f"_depth{str(depth)}"
        
        # Upper branch
        up1 = x
        for i in range(self._hg_num_residual_blocks):
            up1 = self._build_residual_block(up1, f, name=prefix_name+f"_up1_{str(i+1)}")
            
        # Lower branch
        low1 = self._apply_pool(x, k=2, s=2)
        for i in range(self._hg_num_residual_blocks):
            low1 = self._build_residual_block(low1, f, name=prefix_name+f"_low1_{str(i+1)}")
            
        # Recursive
        low2 = None
        if steps_to_go > 1:
            low2 = self._build_hourglass(low1, steps_to_go-1, f, depth=depth+1, name=prefix_name)
        else:
            low2 = low1
            for i in range(self._hg_num_residual_blocks):
                low2 = self._build_residual_block(low2, f, name=prefix_name+f"_low2_{str(i+1)}")
                
        # Additional residual blocks
        low3 = low2
        for i in range(self._hg_num_residual_blocks):
            low3 = self._build_residual_block(low3, f, name=prefix_name+f"_low3_{str(i+1)}")
            
        # Upsample
        up2 = Lambda(lambda x: tf.image.resize_bilinear(x[0], x[1].shape.as_list()[1:3], align_corners=True))([low3, up1])
        
        out = Add()([up1, up2])
        return out
    
    def _build_hourglass_after(self, x_prev, x_now, do_merge=True, name=None):
        prefix_name = name+"_after"
        
        for j in range(self._hg_num_residual_blocks):
            x_now = self._build_residual_block(x_now, self._hg_num_feature_maps, name=prefix_name+f"_after_hg_{str(j+1)}")
        x_now = self._apply_conv(x_now, self._hg_num_feature_maps, k=1, s=1, name=prefix_name)
        x_now = self._apply_bn(x_now, name=prefix_name+"_BatchNorm")
        x_now = Activation('relu')(x_now)
        
        h = self._apply_conv(x_now, self._hg_num_landmarks, k=1, s=1, name=prefix_name+"_hmap")
        
        x_next = x_now
        if do_merge:
            prefix_name = name
            x_hmaps = self._apply_conv(h, self._hg_num_feature_maps, k=1, s=1, name=prefix_name+"_merge_h")
            x_now = self._apply_conv(x_now, self._hg_num_feature_maps, k=1, s=1, name=prefix_name+"_merge_x")
            x_add = Add()([x_prev, x_hmaps])
            x_next = Add()([x_next, x_add])
        return x_next, h
    
    @staticmethod
    def _calculate_landmarks(x, beta=5e1):
        def np_softmax(x, axis=1):
            t = np.exp(x)
            a = np.exp(x) / np.sum(t, axis=axis).reshape(-1,1)
            return a
        
        if len(x.shape) < 4:
            x = x[None, ...]
        h, w = x.shape[1:3]
        ref_xs, ref_ys = np.meshgrid(np.linspace(0, 1.0, num=w, endpoint=True),
                                     np.linspace(0, 1.0, num=h, endpoint=True),
                                     indexing='xy')
        ref_xs = np.reshape(ref_xs, [-1, h*w])
        ref_ys = np.reshape(ref_ys, [-1, h*w])

        # Assuming N x 18 x 45 x 75 (NCHW)
        beta = beta
        x = np.transpose(x, (0, 3, 1, 2))
        x = np.reshape(x, [-1, 18, h*w])
        x = np_softmax(beta * x, axis=-1)
        lmrk_xs = np.sum(ref_xs * x, axis=2)
        lmrk_ys = np.sum(ref_ys * x, axis=2)

        # Return to actual coordinates ranges
        return np.stack([lmrk_xs * (w - 1.0) + 0.5, lmrk_ys * (h - 1.0) + 0.5], axis=2)  # N x 18 x 2
