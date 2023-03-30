# # first, train for H36M
# python main/main.py --cfg ./configs/simple3dmesh_train/baseline_h36m.yml --experiment_name simple3dmesh_train/baseline_h36m --gpus 4

# second, train on PW3D training set for PW3D
python main/main.py --cfg ./configs/simple3dmesh_train/baseline_pw3d.yml --experiment_name simple3dmesh_train/baseline_pw3d --gpus 4 --resume_training
