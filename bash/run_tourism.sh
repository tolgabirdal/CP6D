#!/bin/bash

python main.py -r ~/workspace/dataset/ -d PhotoTourism -l ./dataset/mstransformer_results/PhotoTourism/ -s brandenburg_gate -f False --exp Translation_top15
python main.py -r ~/workspace/dataset/ -d PhotoTourism -l ./dataset/mstransformer_results/PhotoTourism/ -s buckingham_palace -f False --exp Translation_top15
python main.py -r ~/workspace/dataset/ -d PhotoTourism -l ./dataset/mstransformer_results/PhotoTourism/ -s colosseum_exterior -f False --exp Translation_top15
python main.py -r ~/workspace/dataset/ -d PhotoTourism -l ./dataset/mstransformer_results/PhotoTourism/ -s grand_place_brussels -f False --exp Translation_top15
python main.py -r ~/workspace/dataset/ -d PhotoTourism -l ./dataset/mstransformer_results/PhotoTourism/ -s notre_dame_front_facade -f False --exp Translation_top15
python main.py -r ~/workspace/dataset/ -d PhotoTourism -l ./dataset/mstransformer_results/PhotoTourism/ -s palace_of_westminster -f False --exp Translation_top15
python main.py -r ~/workspace/dataset/ -d PhotoTourism -l ./dataset/mstransformer_results/PhotoTourism/ -s pantheon_exterior -f False --exp Translation_top15
python main.py -r ~/workspace/dataset/ -d PhotoTourism -l ./dataset/mstransformer_results/PhotoTourism/ -s taj_mahal -f False --exp Translation_top15
python main.py -r ~/workspace/dataset/ -d PhotoTourism -l ./dataset/mstransformer_results/PhotoTourism/ -s temple_nara_japan -f False --exp Translation_top15
python main.py -r ~/workspace/dataset/ -d PhotoTourism -l ./dataset/mstransformer_results/PhotoTourism/ -s trevi_fountain -f False --exp Translation_top15