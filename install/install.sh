#!/bin/bash

apt install curl
curl -fsSL https://artifacts.elastic.co/GPG-KEY-elasticsearch | apt-key add -
echo "deb https://artifacts.elastic.co/packages/7.x/apt stable main" | tee -a /etc/apt/sources.list.d/elastic-7.x.list
apt update
apt install elasticsearch

cp -f ./install/elasticsearch.yml /etc/elasticsearch/elasticsearch.yml
cp -f ./install/stop_words.txt /etc/elasticsearch/stop_words.txt
apt install systemd
service elasticsearch start
systemctl enable elasticsearch
/usr/share/elasticsearch/bin/elasticsearch-plugin install analysis-nori

curl -XDELETE localhost:9200/wikipedia_contexts
pip install elasticsearch
python ./install/elastic_init.py