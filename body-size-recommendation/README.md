# Body Size Recommendation Model

키(Height)와 몸무게(Weight)를 입력받아 신체 치수를 예측하는 머신러닝 모델과  
이를 API 서버로 제공하여 Spring 웹 서비스와 연동한 프로젝트입니다.

---

## 📌 Project Structure

```
body_size_recommendation/
├── body_size_regression.ipynb   # 데이터 분석 및 모델 학습
├── body_size_regression.py      # 학습 코드 정리본
├── model.pkl                    # 학습 완료된 모델
└── fast_api.py                  # 예측 API 서버
```

---

## 🧠 Model Description

- **Input**: Height, Weight
- **Output**:
  - Arm length
  - Shoulder width
  - Chest
  - Waist
  - Thigh
  - Bottom
  - Hem

OLS Regression 기반 다중 회귀 모델을 사용하여 신체 치수를 예측합니다.

---

## 🚀 How to Run API Server

### 1️⃣ 패키지 설치

```bash
pip install fastapi uvicorn joblib pandas
```

### 2️⃣ 서버 실행

```bash
uvicorn fast_api:app --reload --port 5000
```

서버 실행 후 아래 주소로 예측 요청을 보낼 수 있습니다.

```
POST http://localhost:5000/predict
```

### Request Example

```json
{
  "height": 175,
  "weight": 70
}
```

---

## 🔗 Spring Web Service Integration

Spring(JSP) 회원가입 페이지에서 키/몸무게 입력 시  
아래 API를 호출하여 예측된 신체 치수를 DB에 저장하도록 구성되어 있습니다.

```javascript
url: 'http://localhost:5000/predict'
```

---

## 💡 Purpose

의류 쇼핑몰에서 사용자의 신체 정보를 기반으로  
맞춤 사이즈 추천 기능을 구현하기 위한 모델입니다.
