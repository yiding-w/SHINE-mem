python -m MemoryTest.training.posttrain_shine_teacher_lora \
  --config MemoryTest/config/case_test.yaml \
  --checkpoint-dir /home/wangyiding/SHINE-mem/checkpoints/8gpu_8lora_128metalora_lr5e-5_grouppretrain_1150/train/checkpoint-epoch-1 \
  --teacher-bank-dir MemoryTest/teacher_loras/qa_sft_rank8 \
  --val-file MemoryTest/json_data/splits/semantic_val.json \
  --output-dir MemoryTest/checkpoints/shine_teacher_lora_posttrain_teacher—weight0.2 \
  --fact-counts 4 8 20 \
  --max-steps 2000 \
  --learning-rate 1e-6 \
  --teacher-align-weight 0.2 \
  --answer-weight 0.8 \
  --qa-per-context 4 \
  --context-max-length 1024 \
  --conversation-max-length 512 \
  --answer-max-length 256 \
  --torch-dtype bf16 \
  --use-gradient-checkpoint \
  --freeze-metalora \
  --freeze-mem-tokens

  python -m MemoryTest.training.posttrain_shine_teacher_lora \
  --config MemoryTest/config/case_test.yaml \
  --checkpoint-dir /home/wangyiding/SHINE-mem/checkpoints/8gpu_8lora_128metalora_lr5e-5_grouppretrain_1150/train/checkpoint-epoch-1 \
  --teacher-bank-dir MemoryTest/teacher_loras/qa_sft_rank8 \
  --val-file MemoryTest/json_data/splits/semantic_val.json \
  --output-dir MemoryTest/checkpoints/shine_teacher_lora_posttrain_teacher—weight0 \
  --fact-counts 4 8 20 \
  --max-steps 2000 \
  --learning-rate 1e-6 \
  --teacher-align-weight 0.0 \
  --answer-weight 1.0 \
  --qa-per-context 4 \
  --context-max-length 1024 \
  --conversation-max-length 512 \
  --answer-max-length 256 \
  --torch-dtype bf16 \
  --use-gradient-checkpoint \
  --freeze-metalora \
  --freeze-mem-tokens

python -m MemoryTest.evaluation.eval_shine_memory \
  --config MemoryTest/config/case_test.yaml \
  --checkpoint-dir MemoryTest/checkpoints/shine_teacher_lora_posttrain_teacher—weight0/best \
  --test-file MemoryTest/json_data/splits/semantic_test_augmented.json \
  --num-facts-list 1 2 4 8 12 20 \
  --num-trials 10 \
  --output MemoryTest/results/shine_teacher_lora_posttrain_teacher—weight0_eval.json

python -m MemoryTest.evaluation.eval_shine_memory \
  --config MemoryTest/config/case_test.yaml \
  --checkpoint-dir MemoryTest/checkpoints/shine_teacher_lora_posttrain_teacher—weight0.2/best \
  --test-file MemoryTest/json_data/splits/semantic_test_augmented.json \
  --num-facts-list 1 2 4 8 12 20 \
  --num-trials 10 \
  --output MemoryTest/results/shine_teacher_lora_posttrain_teacher—weight0.2_eval.json