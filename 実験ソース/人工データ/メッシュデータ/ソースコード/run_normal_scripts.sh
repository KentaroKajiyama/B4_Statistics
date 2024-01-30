#!/bin/zsh

python3 k_means_artMesh_normal-1.py & 
python3 k_means_artMesh_normal-1-k4.py & 
python3 k_means_artMesh_normal-2.py &
python3 k_means_artMesh_normal-2-k4.py &
python3 k_means_artMesh_normal-4-k5.py &
python3 k_medians_artMesh_normal-1.py &
python3 k_medians_artMesh_normal-2.py &
python3 k_medians_artMesh_normal-1-k4.py &
python3 k_medians_artMesh_normal-2-k4.py &
python3 k_medians_artMesh_normal-4-k5.py 