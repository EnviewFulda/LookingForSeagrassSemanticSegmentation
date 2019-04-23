import tensorflow as tf
import nnUtils as util
from nets.deeplabV3Plus import model, common

STRIDE = 2
K = 3
initDepth = 128
outputStride = 8 # or 16
backBone = "resnet_v2_50" # or "resnet_v2_101"
batchNormDecay = 0.9997
preTrainedModel = None

def net(image, classes, MODE):    
    #encoding - convolution/pooling
    # Note: arg_scope is optional for inference.
    print("INPUTSHAPE: ", image.get_shape())

    
   # deeplabv3plus with xception
    
    model_options = common.ModelOptions(
      outputs_to_num_classes={"semantic":classes},
      crop_size=[256,512],
      atrous_rates=[12,24,36], #  with OS of 16 -> [6,12,18]
      output_stride=outputStride)
    
    L = model.multi_scale_logits(
      image,
      model_options=model_options,
      image_pyramid=[1.0], # scales 
      weight_decay= 0.00004,
      is_training=True if MODE == "train" else False,
      fine_tune_batch_norm=True)

    softmax = tf.nn.softmax(L["semantic"]["merged_logits"])
    
    return L["semantic"]["merged_logits"], tf.argmax(softmax, axis=3), softmax
