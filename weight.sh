mkdir weights
cd weights
wget http://rpg.ifi.uzh.ch/datasets/netvlad/vd16_pitts30k_conv5_3_vlad_preL2_intra_white.zip
unzip vd16_pitts30k_conv5_3_vlad_preL2_intra_white.zip
rm -r vd16_pitts30k_conv5_3_vlad_preL2_intra_white.zip

cd ..
cd Mask_RCNN
wget https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5

cd ..

cd hfnet
mkdir model
cd model
wget http://robotics.ethz.ch/~asl-datasets/2019_CVPR_hierarchical_localization/hfnet_tf.tar.gz
tar -xvzf hfnet_tf.tar.gz
rm -r hfnet_tf.tar.gz


