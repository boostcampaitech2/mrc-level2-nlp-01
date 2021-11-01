# Readme

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
- sh install/install.sh
- install/elasticsearch_load.ipynb처럼 사용하면 된다.

- configs 폴더 안에 원하는 설정파일을 만든다. (ex. elasticsearch.yaml)
- python main.py --config-name elasticsearch