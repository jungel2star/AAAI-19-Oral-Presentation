import logging
import os
import random
import sys
from collections import deque
from collections import OrderedDict

import click
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from tqdm import tqdm

import adda
from sklearn.cluster import KMeans

sys.path.append("$PWD:$PYTHONPATH")


@click.command()
@click.argument('dataset')
@click.argument('split')
@click.argument('model')
@click.argument('output')
@click.option('--gpu', default='0')
@click.option('--iterations', default=20000)
@click.option('--batch_size', default=50)
@click.option('--display', default=10)
@click.option('--lr', default=1e-4)
@click.option('--stepsize', type=int)
@click.option('--snapshot', default=5000)
@click.option('--netvladflag', type=int)
@click.option('--poolinglayer_mode', type=int,default=2)
@click.option('--pretrainflag', type=int)
@click.option('--cluster_number', type=int,default=32)
@click.option('--weights')
@click.option('--weights_end')
@click.option('--ignore_label', type=int)
@click.option('--solver', default='Adam')
@click.option('--seed', type=int)
def main(dataset, split, model, output, gpu, iterations, batch_size, display,
         lr, stepsize, snapshot, weights, weights_end, ignore_label, solver,
         seed, netvladflag,pretrainflag,poolinglayer_mode,cluster_number):
    adda.util.config_logging()
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        logging.info('CUDA_VISIBLE_DEVICES specified, ignoring --gpu flag')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    logging.info('Using GPU {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))

    if dataset=="art" or dataset=="realworld" or dataset=="clipart" or dataset=="product":
        print("____using officehome dataset:",dataset)
        class_number=65
        center_epoch=80
    else:
        print("____using office31 dataset:",dataset)
        class_number=31
        center_epoch = 80
    vlad_fcn_flag=False
    vlad_WB_flag=False
    if poolinglayer_mode==0:
        vladfinetune_layer=["conv4_3","conv5_1"]
    elif poolinglayer_mode==1:
        vladfinetune_layer = ["conv5_1","conv5_2"]
    elif poolinglayer_mode == 2:
        vladfinetune_layer = ["conv5_2","conv5_3"]
    elif poolinglayer_mode==3:
        vladfinetune_layer = ["conv5_3","fc6"]
    elif poolinglayer_mode==4:
        vladfinetune_layer = ["conv5_3","fc7"]
    else:
        print ("______wrong poolinglayer_mode_____ ")
        vladfinetune_layer = ["conv5_2","conv5_3"]


    def average_gradients(tower_grads):
        """Calculate the average gradient for each shared variable across all towers.
        Note that this function provides a synchronization point across all towers.
        Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
          is over individual gradients. The inner list is over the gradient
          calculation for each tower.
        Returns:
         List of pairs of (gradient, variable) where the gradient has been averaged
         across all towers.
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def inference(netvladflag,im_batch,label_batch,model_fn,vlad_centers_variable,vlad_W,vlad_B):
        net, layers = model_fn(im_batch, scope='vgg_16',class_number=class_number,dropout_ratio=0.5)
        class_loss = tf.losses.sparse_softmax_cross_entropy(label_batch, net)
        correct_prediction = tf.equal(tf.argmax(net, -1), tf.cast(label_batch, dtype="int64"))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
        loss = class_loss
        if netvladflag == 0:
            return accuracy, loss
        else:
            netvlad_alpha = 5000.0
            l2_norm_flag = True
            if netvladflag > 0:
                n_clusters = cluster_number
            else:
                n_clusters = 1
            weights_sparse = 0.0
            vlad_layer = vladfinetune_layer[1]
            print("__________vald____________")
            with tf.variable_scope('NetVLAD'):
                if vlad_WB_flag==False:
                    vlad_W = tf.expand_dims(tf.expand_dims(tf.transpose(vlad_centers_variable) * 2 * netvlad_alpha, axis=0),
                                           axis=0)
                    vlad_B = tf.reduce_sum(tf.square(vlad_centers_variable), axis=1) * (netvlad_alpha) * (-1)
                vlad_input = layers[vlad_layer]
                vlad_rep_output, assgns, loss_vlad_sparse, vlad_centers = \
                    adda.util.netvlad(vlad_input, vlad_centers=vlad_centers_variable, scope="vladconv",
                                      vlad_W=vlad_W, vlad_B=vlad_B, netvlad_alpha=netvlad_alpha,
                                      netvlad_initCenters=n_clusters,
                                      l2_norm_flag=l2_norm_flag)

                #vlad_rep_output = tf.nn.dropout(vlad_rep_output, keep_prob=0.8)
                if vlad_fcn_flag==True:
                    vlad_rep_output = slim.fully_connected(vlad_rep_output, 256, scope='vladfcn1')
                vlad_rep_output = tf.nn.dropout(vlad_rep_output, keep_prob=0.5)
                assgns_arg = tf.cast(tf.argmax(assgns, axis=3), "float32")
                assgns_arg = tf.reshape(assgns_arg, [int(assgns_arg.get_shape()[0]), -1])
                net_vlad = slim.fully_connected(vlad_rep_output, class_number, activation_fn=None, scope='vladfcn2')
                class_loss_vlad = tf.losses.sparse_softmax_cross_entropy(label_batch, net_vlad)
                correct_prediction = tf.equal(tf.argmax(net_vlad, -1), tf.cast(label_batch, dtype="int64"))
                accuracy_vlad = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
                loss_vlad = class_loss_vlad + loss_vlad_sparse * weights_sparse
            return accuracy,loss,accuracy_vlad,loss_vlad,loss_vlad_sparse,vlad_input

    with tf.Graph().as_default(), tf.device('/cpu:0'):
        if seed is None:
            seed = random.randrange(2 ** 32 - 2)
        logging.info('Using random seed {}'.format(seed))
        random.seed(seed)
        np.random.seed(seed + 1)
        tf.set_random_seed(seed + 2)
        dataset_name = dataset
        split_name = split
        dataset = getattr(adda.data.get_dataset(dataset), split)
        model_fn = adda.models.get_model_fn(model)
        im, label = dataset.tf_ops()
        print("____---------_im",im)
        im = adda.models.preprocessing(im, model_fn)
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        lr_var = tf.Variable(lr, name='learning_rate', trainable=False)
        if netvladflag>0:
            n_clusters = cluster_number
        else:
            n_clusters=1

        vlad_channel_num = 512
        netvlad_alpha=5000.0
        with tf.variable_scope('NetVLAD'):
            cluster_centers = np.random.normal(size=(n_clusters, vlad_channel_num), loc=30.0, scale=10.0, )
            vlad_centers_variable = slim.model_variable(
                'centers_vlad',
                shape=cluster_centers.shape,
                initializer=tf.constant_initializer(cluster_centers))

            if vlad_WB_flag==True:
                vlad_W = slim.model_variable( 'vlad_W',
                        shape=(1, 1,) + cluster_centers.transpose().shape,
                        initializer=tf.constant_initializer(
                            cluster_centers.transpose()[np.newaxis, np.newaxis, ...] *
                            2 * netvlad_alpha))
                vlad_B = slim.model_variable(
                        'vlad_B',
                        shape=cluster_centers.shape[0],
                        initializer=tf.constant_initializer(
                            -netvlad_alpha *
                            np.sum(np.square(cluster_centers), axis=1)))
            else:
                vlad_W=None
                vlad_B=None


        optimizer = tf.train.AdamOptimizer(lr_var, 0.5)
        #optimizer = tf.train.MomentumOptimizer(lr_var, 0.99)

        optimizer_fcnonly = tf.train.AdamOptimizer(lr_var*15.0, 0.5)
        optimizer_vladonly = tf.train.AdamOptimizer(lr_var*30.0, 0.5)
        optimizer_vladall = tf.train.AdamOptimizer(lr_var*10.0, 0.5)
        if vlad_fcn_flag==False:
            optimizer_vladonly = tf.train.AdamOptimizer(0.01, 0.5)

        tower_grads_source = []
        tower_grads_source_fcnonly = []
        tower_grads_vladfcnonly = []
        tower_grads_vladall = []

        accuracy=[]
        loss=[]
        accuracy_vlad=[]
        loss_vlad=[]
        loss_spasity=[]
        gpu_visible = [0,1]
        with tf.variable_scope(tf.get_variable_scope()):
            for gpui in range(len(gpu_visible)):
                with tf.device('/gpu:%d' % gpu_visible[gpui]):
                    #with tf.variable_scope('Tower_%d' % (gpu_visible[gpui]))
                    with tf.name_scope('Tower_%d' % (gpu_visible[gpui])) as scope:
                        im_batch, label_batch = tf.train.batch([im, label], batch_size=batch_size)
                        if netvladflag>0:
                            accuracy_temp, loss_temp, accuracy_vlad_temp, loss_vlad_temp, loss_vlad_sparse_temp, vlad_input\
                                =inference(netvladflag, im_batch, label_batch,model_fn,vlad_centers_variable,vlad_W,vlad_B)
                        else:
                            accuracy_temp, loss_temp \
                                = inference(netvladflag, im_batch, label_batch, model_fn, vlad_centers_variable,vlad_W,vlad_B)
                            accuracy_vlad_temp=accuracy_temp
                            loss_vlad_temp=loss_temp
                            loss_vlad_sparse_temp=loss_temp
                        train_variables = tf.trainable_variables()

                        if pretrainflag > 0:
                            if netvladflag>0:
                                source_variables = [v for v in train_variables if ((("vgg_16" in v.name) and
                                    (("conv4" in v.name) or ("conv5" in v.name)   or ("fc" in v.name))) or
                                        (("NetVLAD" in v.name) and ("centers_vladdfadfdsfd" not in v.name)))]
                            else:
                                source_variables = [v for v in train_variables if ((("vgg_16" in v.name) and
                                                (("conv5" in v.name) or ("fc" in v.name))))]

                            print ("____pritraining______")
                            grads_source_temp = optimizer.compute_gradients(loss_temp + 0.0*loss_vlad_temp, var_list=source_variables)
                        else:
                            source_variables = [v for v in train_variables if ((("vgg_16" in v.name) and
                               ((vladfinetune_layer[0] in v.name) or (vladfinetune_layer[1] in v.name) or ("fc" in v.name) ))
                              or( ("NetVLAD" in v.name) and ("centers_vladfdsfsf" not in v.name)))]
                            print("____vlad training______")
                            grads_source_temp = optimizer.compute_gradients(loss_vlad_temp + 0.0*loss_temp, var_list=source_variables)

                        if netvladflag>0:
                            source_variables_fcnonly = [v for v in train_variables if (("vgg_16" in v.name) and
                                                    (("fc" in v.name)))]
                        else:

                            source_variables_fcnonly = [v for v in train_variables if (("vgg_16" in v.name) and
                                    (("fc8" in v.name)  ))]

                        vladfcn_variables= [v for v in train_variables if ("vladfcn" in v.name) ]
                        #vladall_variables =  [v for v in train_variables if ("centers_vlad" in v.name)  ]
                        tf.get_variable_scope().reuse_variables()

                        grads_source_temp_fcnonly = optimizer_fcnonly.compute_gradients(loss_temp, var_list=source_variables_fcnonly)
                        if netvladflag > 0:
                            grads_vladfcntemp = optimizer_vladonly.compute_gradients(loss_vlad_temp, var_list=vladfcn_variables)
                            grads_vladalltemp = optimizer_vladonly.compute_gradients(loss_vlad_temp,
                                                                                     var_list=vladfcn_variables)
                            tower_grads_vladfcnonly.append(grads_vladfcntemp)
                            tower_grads_vladall.append(grads_vladalltemp)


                        tower_grads_source.append(grads_source_temp)
                        tower_grads_source_fcnonly.append(grads_source_temp_fcnonly)

                        loss.append(loss_temp)
                        accuracy.append(accuracy_temp)

                        loss_vlad.append(loss_vlad_temp)
                        accuracy_vlad.append(accuracy_vlad_temp)
                        loss_spasity.append(loss_vlad_sparse_temp)


        grads_source = average_gradients(tower_grads_source)
        grads_source_fcnonly = average_gradients(tower_grads_source_fcnonly)
        grads_vladfcn=average_gradients(tower_grads_vladfcnonly)
        grads_vladall = average_gradients(tower_grads_vladall)
        mapping_step = optimizer.apply_gradients(grads_source, global_step=global_step)
        mapping_step_fcnonly = optimizer_fcnonly.apply_gradients(grads_source_fcnonly, global_step=global_step)


        if netvladflag > 0:
            mapping_step_vladfcnonly = optimizer_vladonly.apply_gradients(grads_vladfcn, global_step=global_step)
            mapping_step_vladall = optimizer_vladonly.apply_gradients(grads_vladall, global_step=global_step)

        loss = tf.reduce_mean(loss)
        loss_vlad=tf.reduce_mean(loss_vlad)
        accuracy=tf.reduce_mean(accuracy)
        accuracy_vlad = tf.reduce_mean(accuracy_vlad)
        loss_spasity = tf.reduce_mean(loss_spasity)

        var_dict_vlad_only = adda.util.collect_vars("NetVLAD")
        var_dict_vgg16 = adda.util.collect_vars('vgg_16')
        var_dict_all = var_dict_vgg16.copy()
        var_dict_all.update(var_dict_vlad_only)
        train_variables = tf.trainable_variables()
        if netvladflag>0:
            vgg16conv_variables = [v for v in train_variables if (("vgg_16" in v.name) and ("conv" in v.name))]
        else:
            vgg16conv_variables = [v for v in train_variables if (("vgg_16" in v.name) and ("fc8" not in v.name))]
        init = tf.global_variables_initializer()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        sess.run(init)

        if weights and pretrainflag==True:
            if netvladflag==True:
                saver1 = tf.train.Saver(var_list=vgg16conv_variables,reshape=True)
            else:
                saver1 = tf.train.Saver(var_list=vgg16conv_variables,reshape=True)

            saver1.restore(sess, weights+'/vgg_16.ckpt')

        if weights and pretrainflag==False:
            weights = tf.train.latest_checkpoint(weights)
            logging.info('Restoring weights from {}:'.format(weights))
            #for src, tgt in var_dict_all.items():
             #   logging.info('    {:30} -> {:30}'.format(src, tgt.name))
            restorer = tf.train.Saver(var_list=var_dict_all,reshape=True)
            restorer.restore(sess, weights)
        saver = tf.train.Saver(var_list=var_dict_all)
        output_dir = os.path.join('snapshot', output)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        losses = deque(maxlen=10)

        bar = tqdm(range(iterations))
        bar.set_description('{} (lr: {:.0e})'.format(output, lr))
        bar.refresh()
        for i in bar:

            if netvladflag > 0:
                if pretrainflag == True:
                    bar_maximum = iterations
                else:
                    bar_maximum = -9000

                if i < bar_maximum:
                    if i <400:
                        _=sess.run([mapping_step_fcnonly])
                    else:
                        _ = sess.run([mapping_step])

                    if i % display == 0:
                        loss_val, accuracy_source_get = \
                            sess.run([loss, accuracy])

                        logging.info('{:10} loss: {:7.4f}  acc: {:7.4f} '
                                     .format('Iteration {}:'.format(i),
                                             loss_val, accuracy_source_get))
                    if i == bar_maximum - 1:
                        layer_conv3_val_total = []
                        for jj in range(center_epoch):
                            if jj % 5 == 0:
                                print("jj:", jj)
                            layer_conv3_normed = tf.reshape(vlad_input,
                                                            [-1, int(vlad_input.get_shape()[-1])])
                            layer_conv3_val = sess.run(layer_conv3_normed)
                            layer_conv3_val_total.append(layer_conv3_val)

                        layer_conv3_val_total = np.reshape(np.array(layer_conv3_val_total), [-1, vlad_channel_num])
                        print("KMeansClustering....")
                        kmeans = KMeans(n_clusters=n_clusters, max_iter=6000, n_init=10).fit(layer_conv3_val_total)
                        centers = np.array(kmeans.cluster_centers_)
                        vlad_W_temp = centers.transpose()[np.newaxis, np.newaxis, ...] *2 *netvlad_alpha
                        vlad_B_temp =np.sum(np.square(centers),axis=1)* (-1)*netvlad_alpha
                        print("vlad_centers_variable assignning....")
                        sess.run(vlad_centers_variable.assign(centers))
                        if vlad_WB_flag==True:
                            sess.run(vlad_W.assign(vlad_W_temp))
                            sess.run(vlad_B.assign(vlad_B_temp))
                        print("vlad_centers_variable assigned....")
                        print("centers[0]", centers[0])
                else:
                    fcn_pretrain_epoch=301
                    if i < fcn_pretrain_epoch:
                        sess.run(mapping_step_vladfcnonly)
                    else:
                        sess.run(mapping_step)

                    if i % display == 0:
                        loss_val, accuracy_val,loss_vlad_val,accuracy_vlad_val,loss_spasity_val = \
                            sess.run([loss, accuracy,loss_vlad,accuracy_vlad,loss_spasity])

                        logging.info(
                            '{:10} loss: {:7.4f}  acc: {:7.4f} loss_vlad: {:7.4f} acc_vlad: {:7.4f} loss_spar: {:7.4f}'
                            .format('Iteration {}:'.format(i),
                                    loss_val,accuracy_val,loss_vlad_val,accuracy_vlad_val,loss_spasity_val))

                        # if accuracy_source_vlad_get>0.95 :
                        #    print("------------weight sparse set 0.1.....")
                        #   sess.run(weights_sparse_var.assign(weights_sparse))

            else:
                if i < 401:
                     _ = sess.run( mapping_step_fcnonly)
                else:
                     _ = sess.run(mapping_step)

                if i % display == 0:
                    loss_val, accuracy_source_get= \
                        sess.run([loss, accuracy])

                    losses.append(loss_val)

                    logging.info('{:20} {:10.4f}   {:10.4f}'
                                 .format('Iteration {}:'.format(i),
                                         loss_val, accuracy_source_get))

            if stepsize is not None and (i + 1) % stepsize == 0:
                lr = sess.run(lr_var.assign(lr * 0.1))
                logging.info('Changed learning rate to {:.0e}'.format(lr))
                bar.set_description('{} (lr: {:.0e})'.format(output, lr))
            if (i + 1) % snapshot == 0:
                snapshot_path = saver.save(sess, os.path.join(output_dir, output),
                                           global_step=i + 1)
                logging.info('Saved snapshot to {}'.format(snapshot_path))

        coord.request_stop()
        coord.join(threads)
        sess.close()

if __name__ == '__main__':
    main()
