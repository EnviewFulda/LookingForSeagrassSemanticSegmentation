import tensorflow as tf
import nnUtils as util
#from nets.deeplabV3 import deeplabV3, deeplabv3PlusResnet
from nets.deeplabV3PlusMobile import model, common

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

    
    # deeplabv3
    #logits = deeplabV3plus.net(image, classes)

    
    
    #deeplabv3plus with resnet
    #network = deeplabv3PlusResnet.deeplab_v3_plus_generator(classes,
     #                                 outputStride,
     #                                 backBone,
     #                                 preTrainedModel,
     #                                 batchNormDecay)

    #L = network(image, True) # True ?= is training
    
    
    
    # deeplabv3plus with xception
    
    model_options = common.ModelOptions(
      outputs_to_num_classes={"semantic":classes},
      crop_size=[256,512], #[256,512],
      atrous_rates=[12,24,36], #  with OS of 16 -> [6,12,18]
      output_stride=outputStride)
    
    L = model.multi_scale_logits(
      image,
      model_options=model_options,
      image_pyramid=[1.0], # scales 
      weight_decay= 0.00004,
      is_training=True if MODE == "train" else False,
      fine_tune_batch_norm=True)

    print("MOBILENETOUTPUT: ", L["semantic"]["merged_logits"].get_shape())
    #decoding - deconvolution/transposing

    #deconv1 = util.deconv(conv4, tf.shape(conv3), [K,K,256,512], "dc1")    
    #deconv2 = util.deconv(deconv1, tf.shape(conv2), [K,K,256,256], "dc2")
    #deconv3 = util.deconv(deconv2, tf.shape(conv1), [K,K,128,256], "dc3") 
    
    #L = util.pixel_dcl(logits, initDepth*2, [K,K], "dc1")
    #L = util.pixel_dcl(L, initDepth*2, [K,K], "dc2")
    #L = util.pixel_dcl(L, initDepth, [K,K], "dc3")
    #L = util.pixel_dcl(L, initDepth, [K,K], "dc4")
    #L = util.pixel_dcl(L, initDepth, [K,K], "dc5")
    
    #L = util.conv(L, [1,1,initDepth,classes], "convEnd", pad="SAME")
    print("OUTPUTSHAPE: ", L["semantic"]["merged_logits"].get_shape())
    
    softmax = tf.nn.softmax(L["semantic"]["merged_logits"])
    
    return L["semantic"]["merged_logits"], tf.argmax(softmax, axis=3), softmax