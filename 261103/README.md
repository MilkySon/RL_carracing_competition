# 261103 experiments

E15_colab_quickstart.ipynb → E16_colab_quickstart.ipynb → 필요 시 E17/E18 메모.
코드: MyDrive/261103/ ; 결과: MyDrive/261103/261103_results/.
첨부 ZIP의 원본 Python 및 requirements는 변경 없이 보존. 최초 실행에만 branch 261103에서 복사. 기존 Drive 코드는 자동 갱신하지 않습니다.
E15: E10 DDQN 일반 replay/seed42/300000 raw frames. E16: 원본 프레임 단위 진단, 재학습 없음.
검증 500–504, 공개 평가 1000–1009, 추가 개발 평가 2000–2029.
학습 seed는 초기화·탐색·트랙을 함께 바꿉니다. E14 CPU 대비 E15 GPU 차이는 통제되지 않은 요인입니다. 엄밀한 비교는 같은 장치·버전으로 재실행.
resume은 replay/RNG 미복원이라 자동 재개하지 않음.
발견: 잔디 조기 종료 없음. 음수 replay는 전체 replay 75%+음수25% 혼합이므로 배치 전체 음수비율은25% 이상일 수 있음.
