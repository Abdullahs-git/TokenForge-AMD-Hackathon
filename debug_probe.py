import router
from eval_public_validation_report import PUBLIC_VALIDATION_TASKS

task = [task for task in PUBLIC_VALIDATION_TASKS if task['task_id'] == 'T03_sentiment_classification'][0]
answer = router.solve_prompt(task['prompt'], '', '', [])
with open('debug_probe_out.txt', 'w', encoding='utf-8') as f:
    f.write('ANSWER_REPR=' + repr(answer) + '\n')
    f.write('EXPECTED_REPR=' + repr(task['simulated_exact_answer'].strip()) + '\n')
    f.write('ANS_STR=' + answer.replace('\n', '\\n') + '\n')
    f.write('EXP_STR=' + task['simulated_exact_answer'].strip().replace('\n', '\\n') + '\n')
