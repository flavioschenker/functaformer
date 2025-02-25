#!/bin/bash
# This script downloads the KITTY360 dataset for LiDAR data

train_list=("2013_05_28_drive_0000_sync"
            "2013_05_28_drive_0002_sync" 
	        "2013_05_28_drive_0003_sync"
            "2013_05_28_drive_0004_sync" 
            "2013_05_28_drive_0005_sync" 
            "2013_05_28_drive_0006_sync" 
            "2013_05_28_drive_0007_sync" 
            "2013_05_28_drive_0009_sync" 
	        "2013_05_28_drive_0010_sync")

DATA_DIR=/your/target/path/to/save/the/zip/files

raw_dir=raw

mkdir -p $DATA_DIR
mkdir -p $DATA_DIR/$raw_dir

cd $DATA_DIR 

# 3d scans
for sequence in ${train_list[@]}; do
    zip_file=${sequence}_velodyne.zip
    wget https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/data_3d_raw/${zip_file}
    unzip -d ${raw_dir} ${zip_file} 
    rm ${zip_file}
done

# timestamps
zip_file=data_timestamps_velodyne.zip
wget https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/data_3d_raw/${zip_file}
unzip -d ${raw_dir} ${zip_file}
rm $zip_file

