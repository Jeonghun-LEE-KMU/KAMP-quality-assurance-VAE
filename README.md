# KAMP AI 경진대회 : 소성가공 품질보증 AI

> **KAMP 스마트제조 AI 경진대회** | 2025.10 | 소성가공 품질보증 이상 탐지

## 대회 개요

| 항목 | 내용 |
|------|------|
| 주최 | 한국스마트제조산업협회 (KAMP) |
| 주제 | 소성가공 공정 품질보증 AI - 제품 불량 탐지 |
| 데이터 | 소성가공 품질보증 AI 데이터셋 (공정 센서 시계열) |
| 평가 지표 | F1-score (정상/불량 이진 분류) |
| 환경 | Python 3.x, TensorFlow/Keras, scikit-learn |

## 문제 정의

소성가공(금속 성형) 공정에서 수집된 센서 데이터를 활용하여 제품의 불량 여부를 탐지하는 이상 탐지(Anomaly Detection) 모델 개발.

- **정상 데이터만으로 학습**하는 비지도/반지도 학습 접근
- 실제 공장 환경의 불균형 데이터 문제 (정상 >> 불량)
- 다수의 공정 센서 피처 활용

## 사용 기술 스택

- **모델**: VAE (Variational Autoencoder) 기반 이상 탐지
- **라이브러리**: `tensorflow`, `keras`, `scikit-learn`, `pandas`, `numpy`
- **전처리**: StandardScaler, 정상 데이터 분리 학습
- **평가**: Reconstruction Error 기반 임계값(threshold) 설정

## 방법론

### VAE (Variational Autoencoder) 기반 이상 탐지

```
정상 데이터 학습
      ↓
Encoder → Latent Space (μ, σ) → Reparameterization
      ↓
Decoder → 재구성 (Reconstruction)
      ↓
재구성 오차 > threshold → 불량 판정
```

**핵심 아이디어**: VAE는 정상 데이터의 분포를 잠재 공간에서 학습. 불량 샘플은 정상 분포에서 멀어 재구성 오차가 크게 발생함.

**모델 구조**
| 레이어 | 구성 |
|--------|------|
| Encoder | Dense(64) → Dense(32) → μ, log_σ² |
| Latent | Reparameterization trick (z = μ + ε·σ) |
| Decoder | Dense(32) → Dense(64) → Dense(input_dim) |
| 활성화 | ReLU + BatchNormalization + Dropout |

## 데이터셋

| 데이터 | 설명 |
|--------|------|
| 소성가공 품질보증 AI 데이터셋 | 메인 품질 분류 데이터 (passorfail 레이블) |
| 열처리 공정최적화 AI 데이터셋 | 공정 파라미터 최적화 참고 |
| 사출성형 공급망최적화 AI 데이터셋 | 공급망 관련 데이터 |

## 프로젝트 구조

```
KAMP/
├── notebooks/
│   └── vae_analysis.ipynb       # VAE 기반 품질 이상 탐지 메인 코드
├── results/                     # 실험 결과 저장 디렉토리
└── README.md
```

> 원본 데이터셋 및 pptx 발표 자료는 용량 문제로 제외.

## 주요 인사이트 및 배운 점

1. **비지도 학습의 강점**: 레이블이 없거나 불균형이 심한 공정 데이터에서 VAE가 강력한 베이스라인을 제공함.

2. **임계값 설정의 중요성**: 재구성 오차의 임계값을 데이터 기반으로 설정하는 것이 F1-score에 결정적인 영향을 미침.

3. **제조 데이터 특성**: 공정 센서 데이터는 시간 순서성과 센서 간 상관관계가 중요함. 피처 엔지니어링 및 EDA가 필수.

4. **정상 데이터만으로 학습**: 불량 데이터 수집 비용이 높은 실제 제조 환경에서 정상 데이터 기반 학습의 실용성 확인.

## 참고 자료

- [KAMP 스마트제조 플랫폼](https://www.kamp-ai.kr/)
- [VAE 논문 (Kingma & Welling, 2013)](https://arxiv.org/abs/1312.6114)
