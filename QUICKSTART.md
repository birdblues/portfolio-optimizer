# NPS-Style Agent Team 포트폴리오 관리 시스템

## 빠른 시작

### 1. 설치
이 폴더의 `.claude/` 디렉토리를 기존 포트폴리오 프로젝트 루트에 복사:

```bash
cp -r .claude/ /path/to/your/portfolio-optimizer/
cp CLAUDE.md /path/to/your/portfolio-optimizer/
```

### 2. Agent Teams 활성화 확인
`.claude/settings.json`에 이미 설정되어 있음:
```json
{ "env": { "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1" } }
```

또는 셸에서 직접:
```bash
export CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1
```

### 3. 사용법

#### SAA 회의 소집 (월 1회)
```
> 줘.
  
  [현재 상황 브리핑 추가]
```

#### TAA 브리핑 (주 1회)
```
> TAA 브리핑 시작. Agent team으로
  quant-analyst와 news-analyst가 협력해서
  views.yaml 업데이트안을 만들고 main.py 실행해줘.
  SAA 위원회를 소집해줘. Agent team을 만들어서
  macro-strategist, regime-analyst, risk-governor가
  현재 시장 레짐을 토론하고 config.yaml 수정안을 의결해
  [금주 이벤트/관심사항 추가]
```

#### 단일 에이전트 활용 (서브에이전트)
```
> quant-analyst 서브에이전트를 써서 현재 EWMA 공분산 상태를 분석해줘.
> news-analyst 서브에이전트로 이번 주 경제 캘린더를 정리해줘.
```

---

## 파일 구조

```
portfolio-optimizer/
├── .claude/
│   ├── settings.json              # Agent Teams 활성화 + 권한
│   ├── agents/
│   │   ├── macro-strategist.md    # SAA: 거시전략 위원 (Opus)
│   │   ├── regime-analyst.md      # SAA: 금융안정성 위원 (Opus)
│   │   ├── risk-governor.md       # SAA: 리스크 관리 위원 (Opus)
│   │   ├── quant-analyst.md       # TAA: 정량분석 운용역 (Sonnet)
│   │   └── news-analyst.md        # TAA: 뉴스/촉매 분석 운용역 (Sonnet)
│   └── skills/
│       ├── saa-session.md         # SAA 회의 프로토콜 (/saa-session)
│       └── taa-briefing.md        # TAA 브리핑 프로토콜 (/taa-briefing)
├── CLAUDE.md                      # 프로젝트 컨텍스트 (모든 에이전트 공유)
├── config.yaml                    # SAA 산출물 (기존 파일)
├── views.yaml                     # TAA 산출물 (기존 파일)
├── views.md                       # 투자 세계관 문서 (기존 파일)
├── main.py                        # BL 실행엔진 (기존 파일)
└── reports/                       # 실행 결과 (기존 디렉토리)
```

---

## 거버넌스 구조

```
┌─ SAA (기금운용위원회) ─────────────────────────┐
│  macro-strategist ↔ regime-analyst ↔ risk-governor │
│  (Opus)           (Opus)          (Opus)          │
│                                                    │
│  산출물: config.yaml (w_mkt, constraints, minsky)  │
│  주기: 월 1회 / 레짐 변화 시                         │
└────────────────────┬───────────────────────────────┘
                     │ config.yaml (읽기 전용)
                     ▼
┌─ TAA (기금운용본부) ───────────────────────────┐
│  quant-analyst ↔ news-analyst                     │
│  (Sonnet)       (Sonnet)                          │
│                                                    │
│  산출물: views.yaml → main.py 실행 → 최종 비중      │
│  주기: 주 1회 / 촉매 발생 시                         │
└────────────────────────────────────────────────────┘
```

핵심 원칙: **TAA는 SAA가 정한 범위 안에서만 작동**

---

## 비용 참고

| 세션 | 에이전트 수 | 모델 | 예상 토큰 | 빈도 |
|------|-----------|------|---------|------|
| SAA 회의 | 3 (+ Lead) | Opus × 3 | 높음 | 월 1회 |
| TAA 브리핑 | 2 (+ Lead) | Sonnet × 2 | 중간 | 주 1회 |
| 단일 서브에이전트 | 1 | Sonnet | 낮음 | 수시 |

비용 절감 팁:
- SAA 소집 전 쟁점을 미리 정리하면 토론 라운드 감소
- 단순 데이터 업데이트는 quant-analyst 서브에이전트 단독 사용
- Agent Team 대신 서브에이전트로도 충분한 경우 구분
