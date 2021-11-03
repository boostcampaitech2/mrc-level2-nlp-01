<div align="center">
  <a href="https://github.com/boostcampaitech2/mrc-level2-nlp-01">
    <img src="https://i.imgur.com/b48hDWD.png" alt="Logo" width="500">
  </a>

  <h3 align="center">Open Domain Question Answering</h3>

  <p align="center">
    Run & Learn Team - BoostCamp AI Second
    <br />
  </p>
</div>

<details>
  <summary>목차 보기</summary>
  <ol>
        <li>
            <a href="#프로젝트 소개">프로젝트 소개</a>
        </li>
        <li>
            <a href="#사용 방법">사용 방법</a>
            <ul>
                <li><a href="#모듈 설치">모듈 설치</a></li>
                <li></li>
            </ul>
        </li>
        <li>
            <a href="#코드 구성">코드 구성</a>
            <ul>
                <li><a href="#built-with">Built With</a></li>
            </ul>
        </li>
  </ol>
</details>

## 프로젝트 소개
<br>

- 우리는 **궁금한 것들이 생겼을 때**, 아주 당연하게 검색엔진을 활용하여 검색을 합니다. 이런 검색엔진은 최근 MRC (기계독해) 기술을 활용하며 매일 발전하고 있습니다.
- 본 대회에서는 우리가 당연하게 활용하던 검색엔진, 그것과 유사한 형태의 시스템을 만들어 볼 것입니다.

<br>
<div align="center">
    <img src="https://i.imgur.com/XE3H9Bp.png" />
</div>
<br>

- 본 ODQA 대회에서 우리가 만들 모델은 two-stage로 구성되어 있습니다. 
    - **Retriever** : 질문과 관련된 문서를 찾는 단계
    - **Reader** : 질문과 문서를 기반으로 답을 찾아내는 단계

<br>
<div align="center">
    <img src="https://i.imgur.com/oKwpFOV.png" />
</div>
<br>

## 설치 단계

### 모듈 설치

```bash=
# konlpy 패키지 설치
sudo apt-get install g++ openjdk-8-jdk python3-dev python3-pip curl

# 필요한 파이썬 패키지 설치. 
pip install -r requirements.txt

# wandb 설정
wandb login

# sigopt 설정
sigopt config
```

### ElasticSeach 설치
```bash=
sh install/elasticsearch/install.sh
```

## 사용 방법

### 코드 구성

- configs : 코드 실행시 설정할 argument를 추가합니다.
- deprecated : 현재 사용되지는 않지만, 추후 사용이 될 코드입니다. (ex. DPR)
- ensemble : 앙상블 코드가 구현되어 있습니다.
- install : 필요한 종속성 프로그램을 설치합니다. (elasticsearch)
- research : 코드 구현전 여러가지 실험이 포함되어 있습니다.
- src : 프로그램 소스코드가 담겨있습니다.
- main.py : 코드 실행 파일입니다.

### Train
#### Feature
- Reader 학습 루트입니다.

#### Usage
- configs 폴더 안에 train_args.example.yaml 을 참고하여 설정파일을 만듭니다. (ex. bert_train.yaml)
- python main.py --config-name [파일이름]
    - 위와 같은 경우 python main.py --config-name bert_train

### 
