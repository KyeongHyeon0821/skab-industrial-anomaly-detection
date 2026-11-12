RF: 정상/이상 라벨로 학습. IF: 학습 파일의 정상 라벨 행만 학습하므로 정상 선별에 라벨 사용.
두 모델 모두 검증 라벨로 설정/임계값 선택. 테스트로 설정을 재선택하지 않음.
Raw/Temporal 4조건 및 임계값 튜닝 전후는 사전에 정한 최종 비교.
최종 선택은 validation_selection.csv, 최종 성능은 test_summary.csv 확인.
3개 시드 평균/표준편차는 학습 난수 변동만 표현. Point adjustment 미적용.
PR-AUC는 Average Precision(AP) 방식. 특정 분할에서의 결과로 일반적인 최우수 모델을 단정하지 않음.
