#!/bin/bash
# Install SC2 and add the custom maps
# Script adapted from https://github.com/oxwhirl/pymarl

export SC2PATH=~/StarCraftII

echo 'StarCraftII is not installed. Installing now ...';
wget --progress=dot:mega http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip
unzip -oP iagreetotheeula SC2.4.10.zip
mv StarCraftII $SC2PATH
rm -rf SC2.4.10.zip

echo 'Adding SMAC maps.'
MAP_DIR="$SC2PATH/Maps/"
echo 'MAP_DIR is set to '$MAP_DIR
mkdir -p $MAP_DIR

wget https://github.com/oxwhirl/smac/releases/download/v0.1-beta1/SMAC_Maps.zip
unzip SMAC_Maps.zip
mv SMAC_Maps $MAP_DIR
rm -rf SMAC_Maps.zip 

echo 'StarCraft II is installed.'

# Install SMAC
pip install -r install_environments/requirements/smacv2.txt