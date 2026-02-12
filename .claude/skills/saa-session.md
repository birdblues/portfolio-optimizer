---
name: saa-session
description: |
  SAA 기금운용위원회 회의를 소집하고 진행하는 프로토콜.
  "SAA 회의", "위원회 소집", "레짐 진단", "중기 배분" 키워드에 반응.
---

# /saa-session — 기금운용위원회 소집 프로토콜

## 개요
국민연금 기금운용위원회를 모델로 한 전략적 자산배분(SAA) 토론 세션.
3명의 위원(macro-strategist, regime-analyst, risk-governor)이 Agent Team으로
독립적 분석 후 상호 토론하여 config.yaml 수정안을 의결합니다.

## 소집 조건
- **정기**: 월 1회 (월초)
- **수시**: 레짐 전환 신호 감지 시, 변동성 급등(VIX > 30) 시, 주요 정책 변경 시

## 진행 절차

### Phase 1: 의제 설정 (Team Lead)
Team Lead는 아래를 정리하여 각 위원에게 배포:
1. 현재 config.yaml 상태 요약
2. 지난 SAA 이후 주요 변화 사항
3. 논의가 필요한 쟁점 (사용자가 제시하거나 자동 감지)

### Phase 2: 독립 분석 (병렬, 각 위원)
각 위원이 자기 프레임워크로 독립 분석:
- **macro-strategist**: 4 Seasons 레짐 판단, 자산군별 방향성
- **regime-analyst**: 민스키 단계, 시스템 리스크, GLD 오버레이 상태
- **risk-governor**: 스트레스 테스트, 제약조건 적정성, drawdown 한도

### Phase 3: 교차 토론 (팀원 간 직접 메시지)
- 각 위원은 다른 위원의 분석 결과를 확인하고 **반론 또는 동의** 표명
- 핵심 쟁점에 대해 **2~3라운드** 토론
- 합의/불일치 사항을 명시적으로 기록

토론 규칙:
1. 주장에는 반드시 **근거(데이터/지표)** 첨부
2. 반론 시 "다만(however)" 구조: 상대 논점 인정 → 보완/반박
3. "틀릴 조건(falsifiable condition)"을 항상 함께 제시
4. 감정적 표현 금지, 정량적 표현 우선

### Phase 4: 의결 (Team Lead = Chairperson)
Team Lead가 종합하여 의결안 작성:

```yaml
# SAA 의결 결과 예시
saa_decision:
  date: "2026-02-10"
  regime: "Late Cycle → Early Contraction"
  minsky_state: "speculative_late"
  
  config_changes:
    w_mkt:
      SPY: 0.35  # 변경 전: 0.40
      TLT: 0.25  # 유지
      GLD: 0.10  # 유지
      DXY: 0.30  # 변경 전: 0.25
    constraints:
      SPY: [0.20, 0.40]  # 상한 45→40 하향
      TLT: [0.10, 0.35]  # 유지
      GLD: [0.07, 0.18]  # 하한/상한 상향
      DXY: [0.15, 0.45]  # 유지
    minsky:
      current_state: "speculative_late"
      gld_targets:
        hedge: 0.10
        speculative: 0.12
        speculative_late: 0.15
        ponzi: 0.20
  
  votes:
    macro-strategist: "찬성 (SPY 하향에 동의, 성장 둔화 반영)"
    regime-analyst: "찬성 (GLD 상향에 동의, 민스키 후반 반영)"  
    risk-governor: "조건부 찬성 (SPY 상한 35%까지 하향 주장했으나 40% 수용)"
  
  dissent: "risk-governor — SPY 상한 40%는 여전히 높다고 판단. 다음 회의에서 재검토 요청."
  
  next_review: "2026-03-10 또는 VIX > 30 발생 시"
  falsifiable: "ISM PMI 50 하회 2개월 연속 시 → Winter 레짐 전환 재검토"
```

### Phase 5: config.yaml 반영
의결 결과를 config.yaml에 반영하는 diff를 생성하되, **실제 수정은 사용자 승인 후**.

## 산출물
1. **의결 결과 YAML** (위 형식)
2. **config.yaml diff** (변경 전/후 비교)
3. **토론 요약**: 각 위원의 핵심 주장과 합의/불일치
4. **다음 회의 의제**: 모니터링 포인트와 재검토 조건

## 비용 관리
- SAA는 Opus 모델 3명 → 토큰 비용 높음
- 사용자가 사전에 쟁점을 정리해두면 효율 극대화
- 명확한 합의 도달 시 조기 종료 가능
