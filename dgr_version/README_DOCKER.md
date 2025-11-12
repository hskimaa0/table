# 표 제목 추출 API - Docker 배포 가이드

## 개요
LLM(Ollama gemma2:2b)을 사용하여 표의 제목을 추출하는 API입니다.

## 구성
- **API 서버**: Flask 기반, 포트 5555
- **Ollama 서버**: LLM 추론, 포트 11434

## 실행 방법

### 1. Docker Compose로 실행 (권장)

```bash
# 1. 빌드 및 실행
docker-compose up -d

# 2. Ollama에 모델 다운로드 (최초 1회)
docker exec ollama ollama pull gemma2:2b

# 3. 로그 확인
docker-compose logs -f

# 4. 중지
docker-compose down
```

### 2. 개별 Docker 실행

```bash
# Ollama 서버
docker run -d --name ollama -p 11434:11434 ollama/ollama:latest
docker exec ollama ollama pull gemma2:2b

# API 서버
docker build -t title-api .
docker run -d --name title-api -p 5555:5555 \
  -e OLLAMA_URL=http://ollama:11434/api/generate \
  --link ollama \
  title-api
```

## API 사용법

### 엔드포인트
- URL: `http://localhost:5555/get_title`
- 메소드: POST
- Content-Type: application/json

### 요청 예시
```json
{
  "tables": [
    {
      "bbox": [100, 200, 500, 400],
      "rows": [...]
    }
  ],
  "texts": [
    {
      "text": "표 3-2 연간 실적",
      "bbox": [100, 150, 300, 170]
    }
  ]
}
```

### 응답 예시
```json
[
  {
    "bbox": [100, 200, 500, 400],
    "rows": [...],
    "title": "표 3-2 연간 실적",
    "title_bbox": [100, 150, 300, 170]
  }
]
```

## 환경변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| OLLAMA_URL | http://localhost:11434/api/generate | Ollama API 엔드포인트 |
| LLM_MODEL | gemma2:2b | 사용할 LLM 모델 |
| LLM_TEMPERATURE | 0.1 | LLM 온도 (0~1) |
| LLM_MAX_TOKENS | 100 | 최대 생성 토큰 수 |

## 디버깅

모든 처리 과정이 콘솔에 출력됩니다:
- Step 1: 후보 수집 (규칙 기반)
- Step 2: 유닛 필터링
- Step 3: 노이즈 필터링
- Step 4: 표 내용 구축
- Step 5: LLM 선택

로그 확인:
```bash
docker-compose logs -f api
```

## 모델 변경

다른 Ollama 모델 사용:
```bash
# 1. 모델 다운로드
docker exec ollama ollama pull qwen2.5:7b

# 2. 환경변수 설정
docker-compose down
# docker-compose.yml에서 LLM_MODEL 수정
docker-compose up -d
```

## 문제 해결

### Ollama 연결 오류
```bash
# Ollama 상태 확인
docker ps | grep ollama
docker exec ollama ollama list

# 재시작
docker-compose restart ollama
```

### API 응답 없음
```bash
# API 로그 확인
docker-compose logs api

# 컨테이너 재시작
docker-compose restart api
```
