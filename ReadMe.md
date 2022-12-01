# Additional steps to be run from within the docker container :
apt-get install zip unzip

## Command to run docker from the tartanvo directory:
nvidia-docker run -it -d --rm --network host --ipc=host -v $PWD:/tartanvo amigoshan/tartanvo:latest

## Command to bring up the container everytime :
docker exec -it <image_id> bash
Ex: docker exec -it fe92847f8e00 bash

## Command to run inference on said camera : 
python vo_trajectory_from_folder.py  --model-name tartanvo_1914.pkl --<nuscenesmini|nuscenes> --batch-size 1 --worker-num 1 --test-dir <path to test image dir> --pose-file <path to pose file>
