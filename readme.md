<p align="center">
    <br>
    <img src="https://i.imgur.com/b48hDWD.png" width="400"/>
    <br>
<p>
<p align="center">
    <a href="https://circleci.com/gh/huggingface/transformers">
        <img alt="Build" src="https://img.shields.io/circleci/build/github/huggingface/transformers/master">
    </a>
    <a href="https://github.com/huggingface/transformers/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/huggingface/transformers.svg?color=blue">
    </a>
    <a href="https://huggingface.co/transformers/index.html">
        <img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/transformers/index.html.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/huggingface/transformers/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/transformers.svg">
    </a>
    <a href="https://github.com/huggingface/transformers/blob/master/CODE_OF_CONDUCT.md">
        <img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg">
    </a>
    <a href="https://zenodo.org/badge/latestdoi/155220641"><img src="https://zenodo.org/badge/155220641.svg" alt="DOI"></a>
</p>

# 목차

[TOC]

## 소개

P stage 3 대회를 위한 베이스라인 코드 

## 설치 방법

### 요구 사항

```
# 필요한 파이썬 패키지 설치. 
pip install -r requirements.txt

# konlpy 패키지 설치
sudo apt-get install g++ openjdk-8-jdk python3-dev python3-pip curl

# wandb 설정
wandb login

# sigopt 설정
sigopt config
```

## 사용방법

### 훈련방법

- configs 폴더 안에 원하는 설정파일을 만든다. (ex. hp_tuning.yaml)
  - 설정파일안에 필요없는걸 지우시면 알아서 기본값을 넣습니다.
- python main.py --config-name hp_tuning

### 엘라스틱 서치 사용
- sh install/elasticsearch/install.sh
- install/elasticsearch/elasticsearch_load.ipynb처럼 사용하면 된다.

- configs 폴더 안에 원하는 설정파일을 만든다. (ex. elasticsearch.yaml)
- python main.py --config-name elasticsearch
