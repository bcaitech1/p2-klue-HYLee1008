# Pstage 02 KLUE Relation extraction

### 개요
관계 추출(Relation Extraction)은 문장의 단어(Entity)에 대한 속성과 관계를 예측하는 문제입니다. 관계 추출은 지식 그래프 구축을 위한 핵심 구성 요소로, 구조화된 검색, 감정 분석, 질문 답변하기, 요약과 같은 자연어처리 응용 프로그램에서 중요합니다. 비구조적인 자연어 문장에서 구조적인 triple을 추출해 정보를 요약하고, 중요한 성분을 핵심적으로 파악할 수 있습니다.

이번 프로젝트에서 주어진 KLUE 데이터를 가지고 주어진 문장의 두 entity 사이의 관계를 추출하는 모델을 만들어 보았습니다. 총 42개의 class가 있으며, 주어진 KLUE 데이터는 label들의 분포가 매우 불균형한 어려운 데이터셋임을 주의해야 합니다.

### 코드 설명
* augmentation_google.py
    * google translation API를 사용하여 부족한 데이터를 증강하기 위한 파일입니다.
    * google translation API를 사용하면, 하루 사용제한이 있고, 실제 구글 번역의 결과와는 다른 성능이 떨어지는 번역 결과가 나와 성능이 좋진 않았습니다.

* augmentation_pororo.py
    * 카카오에서 배포한 pororo를 사용하여 부족한 데이터를 증강하기 위한 파일입니다.
    * pororo 모듈의 pre-trained된 모델을 사용하기 때문에 시간이 매우 오래 걸립니다.

* check_tokenizer.ipynb
    * huggingface에서 가져온 tokenizer가 정상적으로 잘 tokenize하는지 보기 위한 파일입니다.
    * tokenizer에서 발생하는 실수를 줄이기 위해 사용합니다.
    * 모델의 학습 및 추론에는 불필요합니다.

* EDA.ipynb
    * 주어진 KLUE 데이터셋 EDA 파일입니다.
    * 주어진 KLUE 데이터셋의 구조와 분포 등을 간략하게 알 수 있습니다.
    * 모델의 학습 및 추론에는 불필요합니다.

* ensemble.ipynb
    * 앙상블을 수행하는 파일입니다.
    * 여러 모델에서 나온 결과를 이곳에서 앙상블하고, 결과들을 비교할 수 있습니다.

* evaluation.py
    * baseline에 존재하는 코드
    * 쓰지 않는 코드입니다.

* inference_trainer.py
    * huggingface의 trainer를 사용한 high-level inference를 하는 파일입니다.
    * trainer를 사용하여 모델을 학습한 경우 사용하는 파일입니다.

* inference.py
    * 메인 inference 파일입니다.

* load_data.py
    * 데이터 셋을 전처리하고, tokenizer에 집어 넣어 모델에 사용할 수 있게 dataset으로 만드는 파일입니다.
    * 이곳에서 전처리, augmentation같은 데이터 단위 처리들이 이루어집니다.

* loss.py
    * 학습에 사용되는 손실 함수들의 모음 파일입니다.
    * Cross-Entropy Loss, Focal Loss, LabelSmoothing Loss 중 한가지를 선택할 수 있습니다.

* model.py
    * 학습에 사용될 모델들의 모음 파일입니다.
    * huggingface에서 가져온 pre-trained된 모델들을 가져오고, 뒤에 classifier를 붙였습니다.

* train_kFold.py
    * KFold 학습을 하는 파일입니다.

* train_trainer.py
    * huggingface의 trainer를 사용한 high-level train을 하는 파일입니다.

* train.py
    * 메인 학습 파일입니다.

* trainslator.ipynb
    * 데이터 증강을 위한 번역 문장들을 확인하는 파일입니다.
    * 모델의 학습 및 추론에는 불필요합니다.


### 실행 방법
* 학습
```console
$ python train.py
```

* 추론
```console
$ python inference.py
```