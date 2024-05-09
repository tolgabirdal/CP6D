#!/bin/bash

# python main_ori.py -r ~/Data/ -d 7Scenes -l ./dataset/7Scenes_0.5/ -s chess -f False conformal &
# python main_ori.py -r ~/Data/ -d 7Scenes -l ./dataset/7Scenes_0.5/ -s fire -f False conformal &
# python main_ori.py -r ~/Data/ -d 7Scenes -l ./dataset/7Scenes_0.5/ -s office -f False conformal &
# python main_ori.py -r ~/Data/ -d 7Scenes -l ./dataset/7Scenes_0.5/ -s heads -f False conformal &
# python main_ori.py -r ~/Data/ -d 7Scenes -l ./dataset/7Scenes_0.5/ -s pumpkin -f False conformal &
# python main_ori.py -r ~/Data/ -d 7Scenes -l ./dataset/7Scenes_0.5/ -s redkitchen -f False conformal &
# python main_ori.py -r ~/Data/ -d 7Scenes -l ./dataset/7Scenes_0.5/ -s stairs -f False conformal &

python main_ori.py -r ~/Data/ -d CambridgeLandmarks -l ./dataset/CambridgeLandmarks_0.5/ -s KingsCollege -f False conformal &
python main_ori.py -r ~/Data/ -d CambridgeLandmarks -l ./dataset/CambridgeLandmarks_0.5/ -s OldHospital -f False conformal &
python main_ori.py -r ~/Data/ -d CambridgeLandmarks -l ./dataset/CambridgeLandmarks_0.5/ -s ShopFacade -f False conformal &
python main_ori.py -r ~/Data/ -d CambridgeLandmarks -l ./dataset/CambridgeLandmarks_0.5/ -s StMarysChurch -f False conformal &
