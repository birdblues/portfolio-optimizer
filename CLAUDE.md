# Portfolio Optimizer — BL + Minsky Overlay

## 프로젝트 개요
Black-Litterman 모형 + 민스키 오버레이 기반 포트폴리오 최적화 시스템.
국민연금의 SAA(전략적 자산배분)/TAA(전술적 자산배분) 거버넌스 구조를 차용하여
에이전트 팀으로 역할 분담.
멀티 지평(1m/3m/6m) 지원: primary_horizon(기본 3m)으로 최종 비중 결정.

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
  - bl: tau, delta, horizons, primary_horizon
  - ewma: lambda, init_window

views.yaml      ← TAA 관할 (기금운용본부가 결정)
  - views: P, Q_1m/Q_3m/Q_6m, conf_1m/conf_3m/conf_6m 배열
  - assets: 자산 목록

main.py         ← 실행 엔진
  - EWMA 공분산 → 멀티 지평 BL posterior → QP 최적화 → 민스키 오버레이

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
| quant-analyst | EWMA 공분산, BL 정량 분석, 기술적 지표, 지평별 Q/conf 산출 | Sonnet |
| news-analyst | 뉴스 모니터링, 경제지표 해석, 촉매 발굴, 시간지평 태그 | Sonnet |

소집: `/taa-briefing` 또는 "TAA 브리핑"
주기: 주 1회 또는 촉매 발생 시

### 권한 분리 (절대 원칙)
- **TAA는 config.yaml을 수정할 수 없음** (primary_horizon, horizons 포함)
- **TAA는 SAA가 정한 constraints 범위 안에서만 views.yaml 조정**
- **SAA가 config 변경 시 TAA는 즉시 views 재검토 필요**
- **main.py 코드는 누구도 수정하지 않음 — 입력값만 변경**

## BL 모형 핵심 수식 (멀티 지평)
```
# 지평별 공분산 스케일링
Σ_T = T × Σ_daily              # T = 21(1m), 63(3m), 126(6m)

# 각 지평(h)에 대해:
π_h = δ × Σ_T × w_mkt                          # 시장 균형 기대수익률
Ω_h = diag(P × (τΣ_T) × P') × (1/conf_h - 1)  # 뷰 불확실성
μ_BL_h = [(τΣ_T)⁻¹ + P'Ω_h⁻¹P]⁻¹ × [(τΣ_T)⁻¹π_h + P'Ω_h⁻¹Q_h]  # BL posterior
w*_h = argmax(w'μ_BL_h - δ/2 × w'Σ_T w)  s.t. lb ≤ w ≤ ub, Σw = 1

# 최종 비중은 primary_horizon(기본 3m)의 w*에 민스키 오버레이 적용
w_final = minsky_overlay(w*_primary)

# 연환산 변동성
σ_annual = σ_T × √(252/T)
```

## views.yaml 스키마 (멀티 지평)
```yaml
assets: [SPY, TLT, GLD, DXY]
views:
  - id: V1
    name: "뷰 이름"
    P: [1, 0, 0, 0]
    Q_1m: 0.01        # 1개월 기대수익률
    Q_3m: 0.03        # 3개월 기대수익률
    Q_6m: 0.05        # 6개월 기대수익률
    conf_1m: 0.70     # 1개월 확신도 (단기일수록 높을 수 있음)
    conf_3m: 0.60     # 3개월 확신도
    conf_6m: 0.50     # 6개월 확신도 (장기일수록 낮을 수 있음)
```
- 지평별 Q/conf가 없으면 Q_3m, conf 필드로 폴백 (하위호환)

## 민스키 오버레이 상태기계
```
NORMAL(10%) → EUPHORIA(5%) → MARGIN_SHOCK(5%) → BULL_TRAP(5%) → STABILIZE(12-15%) → NORMAL
```
GLD 비중을 상태에 따라 동적 조절. 나머지 자산은 비례 재정규화.
민스키 오버레이는 **primary_horizon의 w*에만 적용**.

## 주의사항
- 헤드라인 서사보다 기저 데이터 우선
- 강제 청산과 전략적 매도 구분
- 시간지평 명시: 1m/3m/6m 뷰가 다를 수 있으며 방향이 상충할 수도 있음
  - 단기(1m): 촉매, 이벤트 드리븐
  - 중기(3m): 구조적 요인, 정책 경로 (primary — 최종 비중 결정)
  - 장기(6m): 추세, 경기사이클 방향
- 관세는 1차 인플레보다 2차 디플레 효과가 클 수 있음
- 정치 발언의 전략적 프레이밍 vs 실제 정책 구분
