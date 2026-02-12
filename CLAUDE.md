# Portfolio Optimizer — BL + Minsky Overlay

## 프로젝트 개요
Black-Litterman 모형 + 민스키 오버레이 기반 포트폴리오 최적화 시스템.
국민연금의 SAA(전략적 자산배분)/TAA(전술적 자산배분) 거버넌스 구조를 차용하여
에이전트 팀으로 역할 분담.

## 자산 유니버스
- **SPY**: S&P 500 ETF (미국 대형주)
- **TLT**: 20년+ 미국 국채 ETF (장기 듀레이션)
- **GLD**: 금 ETF (인플레 헤지 + 테일 리스크 헤지)
- **DXY**: 미국 달러 인덱스 (UUP ETF로 실행)

## 핵심 파일 구조
```
config.yaml     ← SAA 관할 (기금운용위원회가 결정)
  - w_mkt: 시장 균형 비중
  - constraints: 자산별 상하한
  - minsky: current_state, gld_targets
  - bl: tau, delta
  - ewma: lambda, init_window

views.yaml      ← TAA 관할 (기금운용본부가 결정)
  - views: P, Q_3m, conf 배열
  - assets: 자산 목록

main.py         ← 실행 엔진 (수정 불가, 입력만 변경)
  - EWMA 공분산 → BL posterior → QP 최적화 → 민스키 오버레이

views.md        ← 현재 투자 세계관 문서 (참조용)
reports/        ← 실행 결과 저장소
```

## 거버넌스 구조

### SAA Layer (기금운용위원회) — config.yaml 관할
| 에이전트 | 역할 | 모델 |
|---------|------|------|
| macro-strategist | 달리오 4 Seasons, 경기사이클, 자산군 방향성 | Opus |
| regime-analyst | 민스키 사이클, 시스템 리스크, GLD 오버레이 | Opus |
| risk-governor | 허용위험한도, 제약조건, 스트레스 테스트 | Opus |

소집: `/saa-session` 또는 "SAA 회의 소집"
주기: 월 1회 또는 레짐 변화 시

### TAA Layer (기금운용본부) — views.yaml 관할
| 에이전트 | 역할 | 모델 |
|---------|------|------|
| quant-analyst | EWMA 공분산, BL 정량 분석, 기술적 지표 | Sonnet |
| news-analyst | 뉴스 모니터링, 경제지표 해석, 촉매 발굴 | Sonnet |

소집: `/taa-briefing` 또는 "TAA 브리핑"
주기: 주 1회 또는 촉매 발생 시

### 권한 분리 (절대 원칙)
- **TAA는 config.yaml을 수정할 수 없음**
- **TAA는 SAA가 정한 constraints 범위 안에서만 views.yaml 조정**
- **SAA가 config 변경 시 TAA는 즉시 views 재검토 필요**
- **main.py 코드는 누구도 수정하지 않음 — 입력값만 변경**

## BL 모형 핵심 수식
```
π = δ × Σ × w_mkt                          # 시장 균형 기대수익률
Ω = diag(P × (τΣ) × P') × (1/conf - 1)    # 뷰 불확실성
μ_BL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ × [(τΣ)⁻¹π + P'Ω⁻¹Q]  # BL posterior
w* = argmax(w'μ_BL - δ/2 × w'Σw)  s.t. lb ≤ w ≤ ub, Σw = 1
```

## 민스키 오버레이 상태기계
```
NORMAL(10%) → EUPHORIA(5%) → MARGIN_SHOCK(5%) → BULL_TRAP(5%) → STABILIZE(12-15%) → NORMAL
```
GLD 비중을 상태에 따라 동적 조절. 나머지 자산은 비례 재정규화.

## 주의사항
- 헤드라인 서사보다 기저 데이터 우선
- 강제 청산과 전략적 매도 구분
- 시간지평 명시 (1개월/3개월/6개월 view가 다를 수 있음)
- 관세는 1차 인플레보다 2차 디플레 효과가 클 수 있음
- 정치 발언의 전략적 프레이밍 vs 실제 정책 구분
