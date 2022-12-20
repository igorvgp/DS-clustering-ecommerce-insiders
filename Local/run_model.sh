#variable
data=$(date +'%Y-%m-%dT%H:%M:%S')

#path
path='/home/igor/Documents/repos/ds_em_clusterizacao/project/Local'
path_to_envs='/home/igor/.pyenv/versions/3.9.13/envs/ds_em_clusterizacao/bin'

$path_to_envs/papermill $path/src/models/c09_deploy_planning.ipynb $path/reports/c09_deploy_$data.ipynb
