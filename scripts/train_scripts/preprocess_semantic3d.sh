#!/bin/bash
#SBATCH -p cpu 

if [ "$#" -ne 1 ]; then
    echo "Please, provide the path to the preprocessed Semantic3D."
    exit 1
fi

cd ../..
python preprocess_semantic3d.py --dataset_path /export/share/Datasets/Semantic3D \
--out_path /export/share/Datasets/Semantic3D_split