- ~~TrainArgument 설정하기~~
- model은 출력값으로 start_logits과 end_logits을 생성함. 확인결과 각각 (2, 368) 의 크기를 가짐
- inference 를 transformers 의 pipeline을 이용해 생성하기
- main_args.example 문서 보고 기본값으로 바꿔놓기
- wandb 프로젝트 단위로 팀 설정하는 방법 찾기
- retrieval 트레인 만들기
- 데이터 어그멘테이션 방법론 생각하기
- ray 하이퍼파라미터는 hydra, src로 경로설정을 지원안함, 그리고 train dir를 절대경로로 해야하는듯
- 하이퍼파라미터 폴더 생성 안하도록 설정, wandb에 프로젝트 이름 /home/sds같이 안나오게 설정!
- 하이퍼파라미터 리턴타입은 transformers.trainer_utils.BestRun 이다. 그리고 아웃오브메모리 안뜨도록 배치사이즈 설정잘해두자
- hp_space는 이거 참고
```
def default_hp_space_ray(trial) -> Dict[str, float]:
    from .integrations import is_ray_tune_available

    assert is_ray_tune_available(), "This function needs ray installed: `pip " "install ray[tune]`"
    from ray import tune

    return {
        "learning_rate": tune.loguniform(1e-6, 1e-4),
        "num_train_epochs": tune.choice(list(range(1, 6))),
        "seed": tune.uniform(1, 40),
        "per_device_train_batch_size": tune.choice([4, 8, 16, 32, 64]),
    }
```