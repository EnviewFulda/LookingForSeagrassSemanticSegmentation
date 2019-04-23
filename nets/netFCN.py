import tensorflow as tf
import nnUtils as util
import pixelDeconv

STRIDE = 2
K = 3

def net(image, classes, MODE):

    #encoding - convolution/pooling
    conv1 = util.conv(image, [K,K,3,128], "c1", pad="SAME")
    pool1 = util.pool(conv1, 2, STRIDE, name="p1") 
    
    conv2 = util.conv(pool1, [K,K,128,256], "c2", pad="SAME")
    pool2 = util.pool(conv2, 2, STRIDE, name="p2")  
    
    conv3 = util.conv(pool2, [K,K,256,256], "c3", pad="SAME")
    pool3 = util.pool(conv3, 2, STRIDE, name="p2")
    
    conv4 = util.conv(pool3, [K,K, 256, 512], "c4", pad="SAME")
    conv5 = util.conv(conv4, [K,K, 512, 512], "c5", pad="SAME")
    #decoding - deconvolution/transposing

    #deconv1 = util.deconv(conv4, tf.shape(conv3), [K,K,256,512], "dc1")    
    #deconv2 = util.deconv(deconv1, tf.shape(conv2), [K,K,256,256], "dc2")
    #deconv3 = util.deconv(deconv2, tf.shape(conv1), [K,K,128,256], "dc3") 
    
    deconv1 = pixelDeconv.pixel_dcl(conv5, 512, [K,K], "dc1")
    deconv2 = pixelDeconv.pixel_dcl(deconv1, 256, [K,K], "dc2")
    deconv3 = pixelDeconv.pixel_dcl(deconv2, 128, [K,K], "dc3")
    
    conv6 = util.conv(deconv3, [1,1,128,classes], "c6", pad="SAME")

    #deconv3 = tf.image.resize_bilinear(
    #              conv6, [image.get_shape()[1], image.get_shape()[2]], align_corners=True)
    
    softmax = tf.nn.softmax(conv6)

    return conv6, tf.argmax(softmax, axis=3), softmax