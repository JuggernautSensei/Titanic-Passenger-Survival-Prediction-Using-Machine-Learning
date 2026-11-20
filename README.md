<h1 align="center">타이타닉 승객 생존 여부 예측</h1>

<p align="center">Python과 scikit-learn으로 만든 타이타닉 승객 생존 여부 예측</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white" alt="pandas">
</p>

![타이타닉 생존 예측 프로젝트 표지](Resources/Thumbnail.png)

## Contents

* [프로젝트 소개](#프로젝트-소개)
* [결측치 처리](#결측치-처리)
* [탐색적 분석](#탐색적-분석)
* [피처 구성](#피처-구성)
* [모델 비교](#모델-비교)
* [실행](#실행)

## 프로젝트 소개

부경대학교 데이터 사이언스 수업의 텀 프로젝트입니다. [Kaggle Titanic 데이터](https://www.kaggle.com/competitions/titanic/data)로 승객의 생존 여부(`Survived`, 0 또는 1)를 예측하는 이진 분류 문제를 풀었습니다. 학습 데이터는 생존 여부가 기록된 승객 891명, 테스트 데이터는 생존 여부가 없는 승객 418명입니다. 결측치 처리, 탐색적 분석, 피처 구성, 모델 비교, 제출 파일 생성까지 [`Notebook/titanic.ipynb`](Notebook/titanic.ipynb) 하나에서 진행했습니다.

## 결측치 처리

학습 데이터에는 나이 177개, 객실 번호 687개, 탑승항 2개가 비어 있었고, 테스트 데이터에는 나이와 객실 번호 외에 요금 1개가 비어 있었습니다. 열마다 다르게 처리했습니다.

<table width="100%">
  <thead>
    <tr>
      <th align="left">열</th>
      <th align="left">처리</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>Age</code></td>
      <td>학습·테스트 데이터 각각의 중앙값으로 채움</td>
    </tr>
    <tr>
      <td><code>Cabin</code></td>
      <td>891개 중 687개가 비어 있어 열을 제거</td>
    </tr>
    <tr>
      <td><code>Embarked</code></td>
      <td>가장 많이 등장한 탑승항으로 채움</td>
    </tr>
    <tr>
      <td><code>Fare</code></td>
      <td>테스트 데이터의 중앙값으로 채움</td>
    </tr>
  </tbody>
</table>

## 탐색적 분석

![숫자형 열 사이의 상관관계 히트맵. Survived는 Pclass와 -0.34, Fare와 0.26입니다.](Resources/01_CorrelationHeatmap.png)

숫자형 열 중 생존 여부와 상관이 뚜렷한 것은 객실 등급(`Pclass`, -0.34)과 요금(`Fare`, 0.26)이었습니다. 등급은 숫자가 작을수록 높으므로, 높은 등급과 비싼 요금이 생존과 같은 방향으로 움직입니다. 성별과 탑승항은 아직 문자열이라 이 히트맵에는 포함되지 않았습니다.

그래서 문자열 열과 범주형 열은 집단별 생존율로 확인했습니다. `Survived`가 0과 1이므로 집단별 평균이 곧 생존율입니다. 객실 등급, 성별, 형제자매·배우자 수(`SibSp`), 부모·자녀 수(`Parch`)에 대해 각각 "이 값에 따라 생존율이 달라졌을 것"이라는 가정을 세우고 같은 방식으로 비교했습니다.

```python
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
```

![객실 등급과 생존 여부별 나이 분포. 3등급 사망자 쪽 28세 부근에 막대가 크게 솟아 있습니다.](Resources/03_AgeByPclassSurvival.png)

나이는 객실 등급과 생존 여부로 나눠 분포를 봤습니다. 3등급 사망자에서 28세 부근 막대가 유독 높은 것은 비어 있던 나이를 중앙값으로 채운 결과이며, 실제 분포가 아닙니다.

![탑승항별로 객실 등급과 성별에 따른 생존율을 비교한 그래프](Resources/04_EmbarkedPclassSex.png)

탑승항별로 다시 등급과 성별을 나누면, 모든 항구와 등급에서 여성의 생존율이 남성보다 높습니다. 등급에 따른 차이도 항구마다 모양이 다릅니다.

## 피처 구성

모델에 넣을 입력은 분석 결과를 바탕으로 새로 만들거나 합쳤습니다.

이름은 그대로는 쓸 수 없는 문자열이지만, `Mr`, `Mrs`, `Miss`, `Master` 같은 호칭에는 성별, 나이대, 혼인 여부가 함께 담겨 있습니다. 정규식으로 호칭을 뽑아 `Title`을 만들고, 드문 호칭은 `Rare`로, `Mlle`과 `Ms`는 `Miss`로, `Mme`는 `Mrs`로 합쳤습니다.

`SibSp`와 `Parch`는 둘 다 함께 탄 가족 수이므로 하나로 합쳤습니다. 본인을 더한 가족 규모 `FamilySize`와, 혼자 탔는지를 나타내는 `IsAlone`을 만들었습니다.

```python
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
train['IsAlone'] = (train['FamilySize'] == 1).astype(int)
```

나이는 값의 범위를 다섯 구간으로 나눈 `AgeBand`(`pd.cut`)로, 요금은 승객 수가 비슷하게 네 구간으로 나눈 `FareBand`(`pd.qcut`)로 바꿨습니다. 성별과 탑승항은 숫자로 바꾸고, `Name`, `Ticket`, `SibSp`, `Parch`, `Age`, `Fare`는 제거했습니다.

최종 입력은 `Pclass`, `Sex`, `Embarked`, `Title`, `FamilySize`, `IsAlone`, `AgeBand`, `FareBand`의 8개 열입니다.

## 모델 비교

같은 8개 입력으로 다섯 가지 분류 모델을 학습했습니다.

<table width="100%">
  <thead>
    <tr>
      <th align="left">모델</th>
      <th align="right">학습 데이터 정확도</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>결정 트리</td>
      <td align="right">88.66%</td>
    </tr>
    <tr>
      <td>랜덤 포레스트 (트리 100개)</td>
      <td align="right">88.66%</td>
    </tr>
    <tr>
      <td>K-최근접 이웃 (k = 3)</td>
      <td align="right">85.19%</td>
    </tr>
    <tr>
      <td>로지스틱 회귀</td>
      <td align="right">81.03%</td>
    </tr>
    <tr>
      <td>가우시안 나이브 베이즈</td>
      <td align="right">80.13%</td>
    </tr>
  </tbody>
</table>

결정 트리와 랜덤 포레스트가 같은 점수로 가장 높았고, 최종 제출에는 랜덤 포레스트의 예측을 사용했습니다. 테스트 승객의 `PassengerId`와 예측값을 묶어 `submission.csv`로 저장합니다.

다만 이 점수는 학습에 쓴 데이터로 다시 잰 정확도입니다. 결정 트리 계열은 학습 데이터를 거의 그대로 외울 수 있어 이 점수가 높게 나오기 쉽고, 처음 보는 승객에 대한 성능을 뜻하지 않습니다. 검증 데이터를 따로 나누거나 K-겹 교차 검증으로 비교했어야 하며, 노트북의 마지막 정리에도 개선 방향으로 남겼습니다.

## 실행

Python 3, pandas, NumPy, Matplotlib, seaborn, scikit-learn이 필요합니다. Kaggle에서 받은 `train.csv`, `test.csv`, `gender_submission.csv`를 노트북과 같은 폴더에 두고, Jupyter에서 `Notebook/titanic.ipynb`를 위에서부터 실행합니다.
