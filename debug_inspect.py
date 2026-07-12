import router
from eval_public_validation_report import PUBLIC_VALIDATION_TASKS

prompt = [task['prompt'] for task in PUBLIC_VALIDATION_TASKS if task['task_id'] == 'T03_sentiment_classification'][0]
answer = router.solve_prompt(prompt, '', '', [])
expected = [task['simulated_exact_answer'] for task in PUBLIC_VALIDATION_TASKS if task['task_id'] == 'T03_sentiment_classification'][0].strip()
with open('debug_inspect_out.txt', 'w', encoding='utf-8') as f:
    f.write('PROMPT_REPR=' + repr(prompt) + '\n')
    f.write('ANSWER_REPR=' + repr(answer) + '\n')
    f.write('EXPECTED_REPR=' + repr(expected) + '\n')
    f.write('NORM_ANSWER=' + router.__import__('re').sub(r'\\s+', ' ', answer.strip()).lower() + '\n')
    f.write('NORM_EXPECTED=' + router.__import__('re').sub(r'\\s+', ' ', expected.strip()).lower() + '\n')
