
def add_end_idx(answers, contexts):

    answers_ei = []
    for answer, context in zip(answers, contexts):
        ac_answer = answer['text']
        start_idx = answer['answer_start']
        end_idx = start_idx + len(ac_answer)

        if context[start_idx:end_idx] == ac_answer:
            answer['answer_end'] = end_idx
            answers_ei.append(answer)
        else:
            for n in [1, 2]:
                if context[start_idx-n:end_idx-n] == ac_answer:
                    answer['answer_start'] = start_idx - n
                    answer['answer_end'] = end_idx - n
                    answers_ei.append(answer)
    return answers_ei
