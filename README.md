# Talktopia — round-robin / Surface5 비교 실험

하나의 실행 코드에서 두 대화 방식을 선택한다. 실험은 dev에서 실행하고
구현과 가짜 모델 테스트는 기능 브랜치에서 수행한다. 새 결과는 dev의
`outputs/`에 저장한다. 기존 워크트리의 결과와 DB는 이동하거나 덮어쓰지 않는다.

## 설치와 데이터 준비

새 checkout은 Python 3.12와 고정된 `requirements.lock`을 사용한다.

```bash
./install.sh
./load_profiles.sh --voice-source /path/to/reference/voices
.venv/bin/python -m pip install '.[test]'
```

기존 워크트리는 공유 engine과 가상환경을 그대로 사용한다. dev의 두 경로는
main 워크트리를 가리키고, 새 기능 워크트리는 dev의 두 경로를 가리킨다.
공유 engine 파일을 수정하거나 워크트리마다 재설치하지 않는다.

모든 새 실험은 GeminiLight 원본의 다음 데이터를 사용한다.

| 컬렉션 | 개수 |
|---|---:|
| AgentProfile | 40 |
| EnvironmentProfile | 90 |
| RelationshipProfile | 120 |
| EnvAgentComboStorage | 450 |

원본 JSONL 4개의 SHA-256은 `dataset.lock.json`에 고정돼 있다. 기본 원본 경로는
`data/geminilight_sotopia_dataset`이며 `TALKTOPIA_GEMINILIGHT_DATA_DIR`로 지정할 수도 있다.
내용이 다르면 실행을 중단한다. 원본은 읽기 전용으로 사용한다.

실행 DB 기본 경로는 `~/.sotopia/talktopia/geminilight`다. 이전 기본 DB인
`~/.sotopia/talktopia/data`는 그대로 보존한다. 기존 음성 파일을 재사용해
새 DB를 준비하려면 다음처럼 실행한다.

```bash
./load_profiles.sh \
  --data-dir /path/to/geminilight_sotopia_dataset \
  --voice-source ~/.sotopia/talktopia/data/voices
```

실행 전 원본과 DB의 PK·원본 필드를 대조한다. AgentProfile에 추가된 voice ID,
reference WAV 경로·문구는 음성 파일과 대조한다. 개수만 같은 변형 DB도 거부한다.
`TALKTOPIA_DB_DIR`로 다른 경로를 지정해도 새 simulation에는 같은 검증을 적용한다.

## 실행

```bash
# 기본값도 round-robin이다.
./run_pipeline.sh --interaction-mode round-robin
./run_pipeline.sh --interaction-mode surface5-full-duplex
```

`--tag`와 `--reeval-tag`는 사용하지 않는다. 실행 방식·UTC 시각·고유 ID로
디렉터리와 내부 tag를 자동 생성한다. 기본 단일 모델쌍 설정은 기존과 같다.
모델은 `--agent1-model`, `--agent2-model`, `--evaluator-model`로 선택한다.

두 방식은 동일한 순서의 저장 조합 450개를 사용한다. 환경마다 5개 조합이며
agent 역할 순서도 원본대로 유지한다. `--num-envs`, `--pairs-per-env`, `--env-id`,
`--environment-list-pk`, `--use-stored-combos`는 제거했다. 임의 재샘플링을 하지 않는다.

먼저 모델 호출 없이 전체 실행 구성을 준비할 수 있다.

```bash
./run_pipeline.sh --interaction-mode round-robin --dry-run
# 출력된 run의 전체 manifest를 다른 방식에서도 사용한다.
./run_pipeline.sh --interaction-mode surface5-full-duplex \
  --sample-manifest outputs/<round-robin-run>/02_sampled_characters.json --dry-run
```

Manifest는 항상 전체 canonical 집합이어야 한다. 조합 PK·환경 ID·agent 역할 순서,
고유 조합 450개·환경 90개·환경별 5조합·프로필 40개와 episode 순서를 검사한다.
4개 조합을 450행으로 반복하거나 조합이 누락된 입력은 서버 시작 전에 거부한다.

Matrix는 기존 `--agent1-models`와 `--agent2-models` 목록을 사용한다. 예를 들어
4개 모델 × 4개 모델이면 **방식별 450조합 × 16모델쌍 = 7,200 episode**다.
두 방식을 비교할 때 모델 목록·평가 모델·seed·ASR/TTS 설정도 동일하게 지정한다.
고유 환경 수와 총 episode 수는 서로 다른 값이며 보고서에 별도로 기록한다.

## 12턴과 120초

두 방식 모두 최대 **12개 확정 행동**, attempt당 **120초**로 고정한다.
`--max-turns`와 `--episode-timeout-s`로 변경할 수 없다. agent1이 먼저 시작한다.

- `none`과 `backchanneling`은 12턴 예산에서 제외한다.
- speak, hesitation, correction, interruption, 비음성 행동, leave는 각각 1턴이다.
- Backchannel도 원래 행동 번호·음성·대화 기록과 평가 근거에는 남는다.
- 결과의 `turns`는 기존 환경 step/semantic commit 수다. `budget_turns`는 제한에
  포함된 행동 수이며 `action_counts`는 행동별 횟수다. 이 값을 혼동하지 않는다.
- 12번째 예산 행동이 확정되면 추가 행동과 재생을 중단한다. 기존 자연 종료와
  round-robin의 stale 종료는 유지한다.
- 120초는 worker 확보 뒤 대화 시작부터 계산하고 LLM·ASR·TTS 대기를 포함한다.
  대기열과 사후 평가는 제외한다. Timeout은 실패로 기록하며 부분 음성과 진단을
  보존한다. 기존 최대 3회 attempt 정책은 유지하며 재시도를 새 표본으로 세지 않는다.

Round-robin은 `0d7d5ce`의 순차 agent·TTS·상대 ASR·음성 연결 방식을 사용한다.
Surface5는 상대 음성을 듣는 동안 backchannel·수정·끼어들기를 할 수 있다.
이는 대화 시스템 전체를 비교하는 실험으로, backchannel 하나만 바꾼 실험은 아니다.

두 방식의 발화 지시는 40단어다. Round-robin은 기존처럼 프롬프트로 제한하며,
Surface5는 기존 개발 수정대로 검증 상한 50단어를 적용한다. Surface5의 완전한
JSON에서 닫는 code fence가 빠진 경우 허용하는 처리와 DeepSeek answer-only
생성도 유지한다. 잘린 JSON은 허용하지 않는다.

## 작은 검증과 재개

```bash
# 단일 모델쌍에서 450개 manifest를 유지하고 episode 1개만 시도한다.
./run_pipeline.sh --interaction-mode round-robin --episode-limit 1
./run_pipeline.sh --interaction-mode surface5-full-duplex --episode-limit 1

./run_pipeline.sh --resume-run outputs/<run> --episode-limit 1
```

`--episode-limit`는 matrix에서는 **모델쌍마다** 적용된다. 16개 모델쌍에 1을
지정하면 최대 16개 episode를 선택한다. 작은 검증은 단일 모델쌍으로 수행한다.
미실행 episode는 pending으로 남으며 전체 실험 완료로 표시하지 않는다.
전체 simulation 시도가 끝나기 전에는 자동 평가를 시작하지 않는다.

Resume는 저장된 실행 방식·DB 경로·코드/데이터/음성 해시·서버 설정을 검증한다.
과거의 다른 턴 규칙이나 다른 코드로 만든 simulation을 새 실행에 이어 붙이지 않는다.
새 run에는 `experiment`, `database_path`, `dataset_hashes`, `coverage`를 기록한다.

## 평가와 결과 확인

두 방식의 입력 처리는 분리하고, SOTOPIA 평가 모델·프롬프트·7개 차원·temperature와
재시도 정책은 공통으로 사용한다. 상대에게 전달된 음성의 ASR과 확정 행동만
평가한다. Surface5의 생성 원문과 미확정 음성은 점수 근거에 넣지 않는다.

```bash
./run_pipeline.sh --stage reevaluate --simulation-dir outputs/<completed-run>
# 부분 실행에서 완료한 한 episode만 확인할 때는 EpisodeLog를 직접 지정한다.
./run_pipeline.sh --stage reevaluate \
  --episode-json outputs/<run>/simulation/original/episode_0001.json
```

평가는 원본 run의 방식과 DB를 따른다. 과거 run은 저장된 fingerprint에서 DB 경로를
확인한다. 출처를 알 수 없는 단독 EpisodeLog는 `TALKTOPIA_DB_DIR`를 명시해야 하며,
방식도 알 수 없으면 `--interaction-mode`를 명시해야 한다. 새 DB로 과거 평가의
프로필을 자동 대체하지 않는다. 원본 simulation과 별도 평가 디렉터리에 저장한다.

공통 결과는 `run_config.json`, `02_sampled_characters.json`, `03_simulation.json`,
`simulation/original/`, `simulation/readable/`, `simulation/speech/`,
`simulation/audio/`에 있다. Surface5는 `simulation/events/`도 저장한다.
실패 진단은 `simulation/diagnostics/`에서 확인한다.

Matrix의 `matrix_summary.json`과 `matrix_summary.md`에는 실행 방식, 원본 데이터
규모, 모델쌍 수, 예정·완료·실패·미실행 episode 수가 표시된다. 평가 결과의 제외
사례는 0점으로 바꾸지 않고 별도로 보고한다.

## 서버와 개발 검증

`./local_api.sh start|status|stop`으로 서버를 관리한다. 기존 DB를 사용하는 서버는
새 DB용 speech worker와 호환되지 않으므로 실험 중이 아닌지 확인하고 종료한 뒤
새로 시작한다. 서버의 DB 경로와 모델 요청 코드가 현재 dev에 맞아야 한다.
`TALKTOPIA_MODEL_RUNTIME_DIR`와 기존 포트 환경변수로 다른 실행과 분리할 수 있다.

```bash
.venv/bin/python -m pytest -q tests
```

테스트는 임시 DB, 가짜 모델, HTTP mock을 사용한다. 데이터 축소·중복·역할 변경,
두 방식의 턴/timeout, backchannel 겹침, 평가용 ASR 증거, resume와 worker 반환을
검증한다. 일반 수정에서는 [테스트 안내](tests/README.md)에 따라 관련 파일만
실행한다. 공통 실행 코드나 테스트 구성을 수정했을 때와 병합 전에는 전체를
실행한다. 코드 리뷰는 현재 작업 Agent가 직접 수행한다.
