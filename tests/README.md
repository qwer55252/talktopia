# 테스트 실행

프로젝트 설치 후 저장소 루트에서 실행한다. 테스트 의존성은
`.venv/bin/python -m pip install '.[test]'`로 설치한다.

일상적인 수정에서는 변경한 기능과 연결된 테스트 파일을 선택한다. 예를 들어
프롬프트 수정은 다음과 같이 확인한다.

```bash
.venv/bin/python -m pytest -q tests/test_fdb_prompts.py
```

| 변경 내용 | 관련 테스트 파일 (`tests/` 아래) |
| --- | --- |
| LLM 요청·발화 검증 | `test_fdb_requests.py`, `test_models_and_generation.py`, `test_unclosed_json_fence.py` |
| DeepSeek·Qwen 요청 처리 | `test_deepseek_answer_only.py` |
| 백채널 합성·음량 | `test_confirmation_synthesis.py`, `test_nonverbal_backchannel.py` |
| ASR·음성 서버 | `test_speech_pool.py` |
| 실시간 녹음·시각·latency | `test_audio_sample_clock.py`, `test_surface5_timing.py` |
| 대화 실행·종료 | `test_duplex_episode.py`, `test_round_robin_episode.py`, `test_fdb_termination.py`, `test_fdb_single_call_runtime.py`, `test_fdb_passes.py` |
| 모집단·실험 조건·재개 | `test_experiment_population.py`, `test_turn_budget.py`, `test_resume_and_reports.py` |
| 프롬프트·평가 | `test_fdb_prompts.py`, `test_evaluation_modes.py` |

공통 실행 코드를 바꾸거나 테스트 구성을 정리했을 때, 또는 병합 전에는 전체를 확인한다.

```bash
.venv/bin/python -m pytest -q tests
```

같은 검증 코드를 쓰는 입력은 대표 사례를 선택한다. 입력 종류와 행동 종류의
모든 조합을 추가하지 않는다. 실패·취소 시점이나 실제 처리 경로가 다르면
각각 유지한다. 모집단 중복·누락, 12턴·120초 제한, 재시도 횟수, 녹음 누락과
백채널 음량 검증은 실제 실험의 오류를 막는 회귀 테스트다.

LLM·TTS·ASR 호출은 테스트용 응답으로 대체한다. 테스트 통과는 실제 모델의
발화 자연스러움이나 합성 음질을 보장하지 않는다. 이 부분에 영향을 주는 변경은
실제 에피소드의 녹음과 발화·이벤트 기록으로 별도 확인한다.
