#!/bin/bash

apt install curl
curl -fsSL https://artifacts.elastic.co/GPG-KEY-elasticsearch | apt-key add -
echo "deb https://artifacts.elastic.co/packages/7.x/apt stable main" | tee -a /etc/apt/sources.list.d/elastic-7.x.list
apt update
apt install elasticsearch

cp -f ./install/elasticsearch.yml /etc/elasticsearch/elasticsearch.yml
apt install systemd
service elasticsearch start
systemctl enable elasticsearch

pip install elasticsearch
python ./install/elastic_init.py