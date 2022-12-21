#variable
data=$(date +'%Y-%m-%dT%H:%M:%S')

#path
path='/home/ubuntu/DS-clustering-ecommerce-insiders'
path_to_envs='/home/ubuntu/.pyenv/versions/3.9.13/envs/pa005insidersclustering/bin'

$path_to_envs/papermill $path/Deploy/src/models/c09_deploy_planning.ipynb $path/reports/c09_deploy_$data.ipynb
