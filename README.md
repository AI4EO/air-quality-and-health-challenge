# Air Quality and Health Challenge

Welcome to the AI4EO Air Quality and Health Challenge!! 

This repository contains a [Jupyter notebook](starter-pack.ipynb) and [utility functions](utils.py) to get you started exploring and analysing the data. Simply clone the repository or download the files. For a more user-friendly version, visualise the [notebook in nbviewer](https://nbviewer.jupyter.org/github/AI4EO/air-quality-and-health-challenge/blob/main/starter-pack.ipynb) 

## Requirements

In order to run the example notebook you would need to download the training and validation datasets, and set-up the Docker environment provided with read/write permissions to the datasets. To do so, do the following:

 * download the Docker image provided;
 * load the image running from terminal `sudo docker load < ai4eo-public.tar `;
 * run the docker using the utility bash script `sh run_ai4eo_uhost.sh`;
 * attach the container `sudo docker container attach ai4eo-container-uhost`;
 * from within the attached container terminal run JupyterLab `jupyter lab --ip 0.0.0.0 --port 8888 --no-browser`;
 * click on one of the links to JupyterLab home and you are ready to go. 
 
## Help and support

For any issue relating to the notebook or the data, open a ticket in the challenge Forum, so that we and the community can adequately support you. For improvements and fixes ot the notebook, issues and pull requests can be opened in this repository.
 
