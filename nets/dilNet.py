# dilated convolutional NN according to: https://arxiv.org/pdf/1511.07122.pdf
import tensorflow as tf
import nnUtils as util
import pixelDeconv

STRIDE = 2
K = 3

def net(image, classes, MODE):

    # Paper: Context Network
    # conv1 = util.conv(image, [K,K,3,16], "c1", dilation=1, pad="SAME")
    # conv2 = util.conv(conv1, [K,K,16,16], "c2", dilation=1, pad="SAME")
    # conv3 = util.conv(conv2, [K,K,16,32], "c3", dilation=2, pad="SAME")
    # conv4 = util.conv(conv3, [K,K,32,32], "c4", dilation=4, pad="SAME")
    # conv5 = util.conv(conv4, [K,K,32,64], "c5", dilation=8, pad="SAME")
    # conv6 = util.conv(conv5, [K,K,64,64], "c6", dilation=16, pad="SAME")
    # conv7 = util.conv(conv6, [K,K,64,64], "c7", dilation=1, pad="SAME")
    # conv8 = util.conv(conv7, [1,1,64,classes], "c8", dilation=1, pad="SAME")

    # return conv8, tf.argmax(conv8, axis=3)

    # Paper: Front End 

    e1_c1 = util.conv(image, [K, K, 3, 64], "e1_c1")
    e1_c2 = util.conv(e1_c1, [K, K, 64, 64], "e1_c2")
    pool1 = util.pool(e1_c1, 2, STRIDE, name="pool1")

    e2_c1 = util.conv(pool1, [K, K, 64, 128], "e2_c1")
    e2_c2 = util.conv(e2_c1, [K, K, 128, 128], "e2_c2")
    pool2 = util.pool(e2_c2, 2, STRIDE, name="pool2")

    e3_c1 = util.conv(pool2, [K, K, 128, 256], "e3_c1")
    e3_c2 = util.conv(e3_c1, [K, K, 256, 256], "e3_c2")
    e3_c3 = util.conv(e3_c2, [1, 1, 256, 256], "e3_c3")
    pool3 = util.pool(e3_c3, 2, STRIDE, name="pool3")

    e4_c1 = util.conv(pool3, [K, K, 256, 512], "e4_c1", dilation=2)
    e4_c2 = util.conv(e4_c1, [K, K, 512, 512], "e4_c2", dilation=2)
    e4_c3 = util.conv(e4_c2, [1, 1, 512, 512], "e4_c3", dilation=2)
    
    e5_c1 = util.conv(e4_c3, [K, K, 512, 512], "e5_c1", dilation=4)
    e5_c2 = util.conv(e5_c1, [K, K, 512, 512], "e5_c2", dilation=4)
    e5_c3 = util.conv(e5_c2, [1, 1, 512, 512], "e5_c3", dilation=4)

    #de1 = util.deconv(e5_c3, tf.shape(e2_c2), [K,K,512,1024], "de1")
    #de2 = util.deconv(de1, tf.shape(e1_c2), [K,K,128,512], "de2")

    deconv1 = pixelDeconv.pixel_dcl(e5_c3 , 512, [K,K], "dc1")
    deconv2 = pixelDeconv.pixel_dcl(deconv1, 256, [K,K], "dc2")
    deconv3 = pixelDeconv.pixel_dcl(deconv2, 128, [K,K], "dc3")

    
    final = util.conv(deconv3, [K,K, 128,classes], "final")

    softmax = tf.nn.softmax(final)

    return final, tf.argmax(softmax, axis=3), softmax