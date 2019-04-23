import tensorflow as tf
import nnUtils as util
import pixelDeconv

STRIDE = 2

# uNet according to:
# Ronneberger et al. - U-Net: Convolutional Networks for Biomedical Image Segmentation
# https://arxiv.org/pdf/1505.04597.pdf 

def net(image, classes, MODE):

    #image = tf.image.resize_images(image, [101,101])#, method=ResizeMethod.NEAREST_NEIGHBOR)
    bsize = image.get_shape()[0].value
    f = 3 # kernel size
    #tf.summary.scalar("input", tf.reduce_sum(image))

    #encoding - downsampling
    # encoding level 1
    e1_c1 = util.conv(image, [f,f,3,1], "e1_c1", "SAME")
    e1_c2 = util.conv(e1_c1, [f,f,1,64], "e1_c2", "SAME")
    e1_c3 = util.conv(e1_c2, [f,f,64,64], "e1_c3", "SAME")
    pool1 = util.pool(e1_c3, 2, 2, name="pool1")

    #tf.summary.scalar("e1_c1", tf.reduce_sum(e1_c1))


    # encoding level 2
    e2_c1 = util.conv(pool1, [f,f,64,128], "e2_c1", "SAME")
    e2_c2 = util.conv(e2_c1, [f,f,128,128], "e2_c2", "SAME")
    e2_c3 = util.conv(e2_c2, [f,f,128,128], "e2_c3", "SAME")
    pool2 = util.pool(e2_c3, 2, 2, name="pool2")
    
    tf.summary.scalar("e2_c1", tf.reduce_sum(e2_c1))

    # encoding level 3
    e3_c1 = util.conv(pool2, [f,f,128,256], "e3_c1", "SAME")
    e3_c2 = util.conv(e3_c1, [f,f,256,256], "e3_c2", "SAME")
    e3_c3 = util.conv(e3_c2, [f,f,256,256], "e3_c3", "SAME")
    pool3 = util.pool(e3_c3, 2, 2, name="pool3")


    #tf.summary.scalar("e3_c1", tf.reduce_sum(e3_c1))

    # encoding level 4
    e4_c1 = util.conv(pool3, [f,f,256,512], "e4_c1", "SAME")
    e4_c2 = util.conv(e4_c1, [f,f,512,512], "e4_c2", "SAME")
    e4_c3 = util.conv(e4_c2, [f,f,512,512], "e4_c3", "SAME")
    pool4 = util.pool(e4_c3, 2, 2, name="pool4")

    #tf.summary.scalar("e4_c1", tf.reduce_sum(e4_c1))


    # encoding level 5
    e5_c1 = util.conv(pool4, [f,f,512,1024], "e5_c1", "SAME")
    e5_c2 = util.conv(e5_c1, [f,f,1024,1024], "e5_c2", "SAME")
    deOut = [bsize, e5_c2.get_shape()[1].value*STRIDE, e5_c2.get_shape()[2].value*STRIDE, 512]
    #de_dc1 = util.deconv(e5_c2, deOut, [f, f, 512, 1024], "de_dc1")
    de_dc1 = pixelDeconv.pixel_dcl(e5_c2, 512, [f,f], "de_dc1")
    #tf.summary.scalar("e5_c2", tf.reduce_sum(e5_c2))
    #tf.summary.scalar("de_dc1", tf.reduce_sum(de_dc1))


    # decoding - upsampling 
    # decoding level 1   
    sliced = tf.slice(e4_c3, [0,0,0,0],[-1, deOut[1], deOut[2],-1])
    de1_c1 = util.conv(tf.concat([sliced, de_dc1], 3), [f,f,1024,512], "de1_c1", "SAME")
    de1_c2 = util.conv(de1_c1, [f,f,512,512], "de1_c2", "SAME")
    deOut = [bsize, de1_c2.get_shape()[1].value*STRIDE, de1_c2.get_shape()[2].value*STRIDE, 256]
    #de1_dc1 = util.deconv(de1_c2, deOut, [f,f, 256, 512],  "de1_dc1")
    de1_dc1 = pixelDeconv.pixel_dcl(de1_c2, 256, [f,f], "de1_dc1")
    
    # decoding level 2 
    sliced = tf.slice(e3_c3, [0,0,0,0],[-1, deOut[1], deOut[2],-1]) 
    de2_c1 = util.conv(tf.concat([sliced, de1_dc1], 3), [f,f,512,256], "de2_c1", "SAME")
    de2_c2 = util.conv(de2_c1, [f,f,256,256], "de2_c2", "SAME")
    deOut = [bsize, de2_c2.get_shape()[1].value*STRIDE, de2_c2.get_shape()[2].value*STRIDE, 128]
    #de2_dc1 = util.deconv(de2_c2, deOut, [f,f, 128, 256], "de2_dc1")
    de2_dc1 = pixelDeconv.pixel_dcl(de2_c2, 128, [f,f], "de2_dc1")
    
    # decoding level 3 
    sliced = tf.slice(e2_c2, [0,0,0,0],[-1, deOut[1], deOut[2], -1]) 
    de3_c1 = util.conv(tf.concat([sliced, de2_dc1], 3), [f,f,256,128], "de3_c1", "SAME")
    de3_c2 = util.conv(de3_c1, [f,f,128,128], "de3_c2", "SAME")
    deOut = [bsize, de3_c2.get_shape()[1].value*STRIDE, de3_c2.get_shape()[2].value*STRIDE, 64]
    #de3_dc1 = util.deconv(de3_c2,deOut, [f,f, 64, 128],  "de3_dc1")
    de3_dc1 = pixelDeconv.pixel_dcl(de3_c2, 64, [f,f], "de3_dc1")
    
    # decoding level 3 
    sliced = tf.slice(e1_c2, [0,0,0,0],[-1, deOut[1], deOut[2],-1]) 
    de4_c1 = util.conv(tf.concat([sliced, de3_dc1], 3), [f,f,128,64], "de4_c1", "SAME")
    de4_c2 = util.conv(de4_c1, [f,f,64,64], "de4_c2", "SAME")
    de4_c3 = util.conv(de4_c2, [f,f,64,64], "de4_c3", "SAME")
    de4_c4 = util.conv(de4_c3, [f,f,64,64], "de4_c4", "SAME")

    final = util.conv(de4_c4, [1,1,64,classes], "final", "SAME")
    #tf.summary.scalar("final", tf.reduce_sum(final))

    softmax = tf.nn.softmax(final)

    return final, tf.argmax(softmax, axis=3), softmax