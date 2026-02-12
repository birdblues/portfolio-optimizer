---
name: taa-briefing
description: |
  TAA 기금운용본부 전술적 브리핑 프로토콜.
  "TAA 브리핑", "뷰 업데이트", "views.yaml 수정", "리밸런싱" 키워드에 반응.
---

# /taa-briefing — 기금운용본부 전술적 브리핑 프로토콜

## 개요
국민연금 기금운용본부를 모델로 한 전술적 자산배분(TAA) 분석 세션.
2명의 운용역(quant-analyst, news-analyst)이 Agent Team으로
정량/정성 분석을 수행하고 views.yaml 수정안을 산출합니다.
**SAA(config.yaml)가 정한 제약 범위 안에서만** 조정합니다.
멀티 지평(1m/3m/6m)별 Q 및 conf를 산출합니다.

## 소집 조건
- **정기**: 주 1회 (월요일 또는 화요일)
- **수시**: 주요 경제지표 발표 직후, 시장 급변 시, SAA 의결 직후

## SAA → TAA 연결 규칙
TAA는 SAA의 결정을 **반드시 준수**합니다:

| SAA 결정 (config.yaml) | TAA 권한 |
|----------------------|---------|
| w_mkt (균형 비중) | 읽기 전용 — 수정 불가 |
| constraints (상하한) | 읽기 전용 — 수정 불가 |
| minsky.current_state | 읽기 전용 — 수정 불가 |
| minsky.gld_targets | 읽기 전용 — 수정 불가 |
| delta, tau, lambda | 읽기 전용 — 수정 불가 |
| **primary_horizon** | **읽기 전용 — 수정 불가** (SAA 소관) |
| **horizons (1m/3m/6m)** | **읽기 전용 — 수정 불가** (SAA 소관) |

| TAA 결정 (views.yaml) | TAA 권한 |
|---------------------|---------|
| 각 view의 Q_1m, Q_3m, Q_6m | **수정 가능** |
| 각 view의 conf_1m, conf_3m, conf_6m | **수정 가능** |
| view 추가/삭제 | **수정 가능** (메이저 변경 시 기록) |
| view의 rationale | **수정 가능** |

## 진행 절차

### Phase 1: 데이터 수집 (병렬)

**quant-analyst 태스크:**
1. `uv run python main.py` 실행 → 최신 Σ, π, μ_BL (전 지평) 확인
2. 전주 대비 공분산/상관관계 변화 분석
3. 각 자산의 기술적 지표 업데이트
4. 지평별 Q 범위 및 conf 보정 산출

**news-analyst 태스크:**
1. 금주 핵심 이벤트 스캔 (웹 검색 + Supabase DB)
2. 경제지표 서프라이즈 분석
3. 정책 발언 파싱 및 원문 확인
4. 이벤트별 시간지평 태그(1m/3m/6m) 부여

### Phase 2: 교차 검증 (팀원 간 직접 메시지)
- quant → news: "정량적으로 SPY Q_3m +2%는 z-score 0.8. 뉴스 관점에서도 유지?"
- news → quant: "NFP 연기로 불확실성 증가. 1m conf를 낮추고 3m은 유지하는 건?"
- 상충점 식별 및 해결 (또는 양론 병기)
- 지평 간 일관성 점검: 1m 뷰와 3m 뷰의 방향이 상충하면 반드시 논의

### Phase 3: views.yaml 수정안 작성 (TAA Director = Team Lead)

Team Lead가 종합하여 수정안 작성:

```yaml
# TAA 브리핑 결과 예시 (멀티 지평)
taa_decision:
  date: "2026-02-10"
  trigger: "NFP 연기 + 5년 BEI 3.4% 상승 확인"

  views_changes:
    - view: V5 (SPY 절대뷰)
      Q_1m: "+0.80% → +0.60%"
      Q_3m: "+2.00% → +1.50%"
      Q_6m: "+3.50% → +3.00%"
      conf_1m: "60% → 55%"
      conf_3m: "50% → 45%"
      conf_6m: "40% → 35%"
      reason: "NFP 연기로 고용 데이터 불확실성 증가. 전 지평 하향 조정."
      quant_support: "SPY RSI 62, 과열 아님. 50일MA 위 유지."
      news_support: "셧다운 장기화 시 재정지출 지연 리스크."

    - view: V3 (TLT 절대뷰)
      Q_1m: "+0.30% → +0.40%"
      Q_3m: "+0.875% → +1.20%"
      Q_6m: "+2.00% → +2.50%"
      conf_1m: "55% → 60%"
      conf_3m: "45% → 50%"
      conf_6m: "40% → 45%"
      reason: "Q2 국채 발행 축소 확인 + 안전자산 수요 증가."

    - view: V1 (GLD 절대뷰)
      Q_1m: "-0.50% 유지"
      Q_3m: "-1.25% 유지"
      Q_6m: "-2.00% 유지"
      conf: "60% 유지 (전 지평 동일)"
      reason: "민스키 오버레이 상태 변경 없음. SAA 유지."

  dissent:
    quant_analyst: "DXY 6m 뷰 변경에 약한 반대. 정량적 근거 부족."
    resolution: "conf_6m 30%로 낮게 유지하여 영향력 제한."
```

### Phase 4: BL 실행 및 검증

```bash
# 수정된 views.yaml로 main.py 실행
uv run python main.py
```

결과 검증 체크리스트:
- [ ] 모든 비중이 constraints 범위 내
- [ ] 합계 = 100%
- [ ] 민스키 오버레이 후 GLD가 적정 범위
- [ ] 포트폴리오 변동성이 SAA 허용 범위 내
- [ ] 전주 대비 비중 변화가 합리적 (±3% 이내 권장)
- [ ] **지평별 w* 비교 테이블 검토**: 1m/3m/6m 간 비중 차이가 극단적이지 않은지

### Phase 5: 리포트 산출
- BL 실행 리포트 (reports/ 디렉토리에 자동 저장)
- 주간 변화 요약 (지평별 비교 포함)
- 다음 주 모니터링 포인트

## 산출물
1. **views.yaml 수정안** (변경 전/후 + 근거, 전 지평)
2. **main.py 실행 결과** (최종 비중, 변동성, 지평별 비교)
3. **주간 변화 요약**: 지난주 vs 이번 주 비중 변화
4. **다음 주 모니터링**: 예정 이벤트, 트리거 조건 (시간지평 태그 포함)
5. **SAA 에스컬레이션 필요 여부**: 제약 변경이 필요하면 SAA 소집 건의

## 비용 관리
- TAA는 Sonnet 모델 2명 → SAA 대비 비용 효율적
- 단순 데이터 업데이트만 있으면 quant-analyst 단독으로도 가능
- 주요 이벤트 발생 시에만 news-analyst 투입
