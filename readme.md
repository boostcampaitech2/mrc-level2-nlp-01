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
            <a href="#프로젝트_소개">프로젝트 소개</a>
        </li>
        <li>
            <a href="#설치 단계">설치 단계</a>
            <ul>
                <li><a href="#모듈 설치">모듈 설치</a></li>
                <li><a href="#ElasticSearch 설치">ElasticSearch 설치</a></li>
            </ul>
        </li>
        <li>
            <a href="#사용 방법">사용 방법</a>
            <ul>
                <li><a href="#코드 구성">코드 구성</a></li>
                <li><a href="#Train">Train</a></li>
                <li><a href="#Hyperparameter Tune">Hyperparameter Tune</a></li>
                <li><a href="#Retrieval">Retrieval</a></li>
                <li><a href="#Inference">Inference</a></li>
            </ul>
        </li>
  </ol>
</details>

# 프로젝트 소개
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

# 설치 단계

## 모듈 설치

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

## ElasticSeach 설치
```bash=
sh install/elasticsearch/install.sh
```

# 사용 방법

## 코드 구성

### Tree
- configs : 코드 실행시 설정할 argument를 추가합니다.
- deprecated : 현재 사용되지는 않지만, 추후 사용이 될 코드입니다. (ex. DPR)
- ensemble : 앙상블 코드가 구현되어 있습니다.
- install : 필요한 종속성 프로그램을 설치합니다. (elasticsearch)
- research : 코드 구현전 여러가지 실험이 포함되어 있습니다.
- src : 프로그램 소스코드가 담겨있습니다.
- main.py : 코드 실행 파일입니다.

### 사용법

![](https://i.imgur.com/FXfm242.png)
- example.yaml 파일을 통해서 개인 설정파일을 만듭니다.
- 해당 설정파일을 main.py를 통해 실행하면 됩니다.
    - 만약 inf_test.yaml을 파일을 만들었다면 다음과 같이 실행하면 됩니다.
```bash=
python main.py --config-name inf_test
```

## Train

### Feature
![image alt](https://i.imgur.com/kQd8r3F.png)
- Reader 학습 루트이며, 자동으로 wandb에 기록됩니다.
- config/train_args.example.yaml 파일을 통해 옵션을 설정할 수 있습니다.


## Hyperparameter Tune

### Feature

![](https://i.imgur.com/e63LG2W.png)  |  ![](https://i.imgur.com/PUcUGTa.png)
:-------------------------:|:-------------------------:
![](https://i.imgur.com/iazvNwB.png) | ![](https://i.imgur.com/GW3ZIJL.png)

- Sigopt를 사용하여 최적의 하이퍼파라미터를 찾습니다.
- 가장 잘 나온 모델을 검색후 저장합니다.
- config/hp_tunes.example.yaml 파일을 통해 옵션을 설정할 수 있습니다.



## Retrieval
### Feature

<div align="center">
    <img src="https://i.imgur.com/6Qvp1h4.png" width="500" />   <img src="https://i.imgur.com/IaQM7mb.png"  width="400" />
</div>
<br>

- 질문이 담긴 데이터셋을 지문까지 담긴 데이터셋으로 변환합니다.
- top10 을 기준으로 validation 셋을 통해 테스트 하였습니다.
- 속도는 elasticsearch가 가장 빨랐지만, 정확도는 저희가 만든 Okt기반 토크나이저가 가장 정확했습니다.
- 해당 요소는 전부 구현되어 있고, Huggingface에 탑재된 그 외의 토크나이저를 불러올 수 있습니다.

## Inference

- 모델과 Retrieval로 수행된 데이터셋을 불러와서 inference를 시행합니다.
- nbest_predictions와 predictions를 출력하며 앙상블에 이용할 수 있습니다.

---

- 추가 할 것
    - 다양한 실험
    - Ensemble 사용법
    - 실패한 것들
    - 그외 기타 등등라머아ㅓㄴ랴ㅓ재ㅑ덜
