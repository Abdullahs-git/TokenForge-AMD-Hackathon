import router
from eval_public_validation_report import PUBLIC_VALIDATION_TASKS

task = [task for task in PUBLIC_VALIDATION_TASKS if task['task_id'] == 'T03_sentiment_classification'][0]
answer = router.solve_prompt(task['prompt'], '', '', [])
print('ANS_REPR=' + repr(answer))
print('EXP_REPR=' + repr(task['simulated_exact_answer'].strip()))
