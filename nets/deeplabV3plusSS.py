import tensorflow as tf
# import nnUtils as util
# from nets.deeplabV3Plus import model, common

from LookingForSeagrassSemanticSegmentation.nets.deeplabV3Plus import model, common


def net(image, classes, MODE):    
    # Note: arg_scope is optional for inference.
   # deeplabv3plus with xception
    
    model_options = common.ModelOptions(
      outputs_to_num_classes={"semantic":classes},
      crop_size=[256,512],
      atrous_rates=[12,24,36], #  with OS of 16 -> [6,12,18]
      output_stride= 8) # or 16)
    
    L = model.multi_scale_logits(
      image,
      model_options=model_options,
      image_pyramid=[1.0], # scales 
      weight_decay= 0.00004,
      is_training=True if MODE == "train" else False,
      fine_tune_batch_norm=True)

    softmax = tf.nn.softmax(L["semantic"]["merged_logits"])
    
    return L["semantic"]["merged_logits"], tf.argmax(softmax, axis=3), softmax
