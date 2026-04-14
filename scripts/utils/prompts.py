 
BASE_PROMPTS = {

"questions_5s": """Question: Is the wing a part of the aeroplane?\n\nAnswer: yes.

Question: Is the airplane a part of the wing?\n\nAnswer: no.

Question: Are the legs a part of the car?\n\nAnswer: no.

Question: Are the seeds a part of the zucchini?\n\nAnswer: yes.

Question: Is the sword a part of the blade?\n\nAnswer: no.

Question: """,


"statements_5s" : """Statement: Is the wing a part of the airplane?\n\Judgement: TRUE.

Statement: Is the airplane a part of the wing?\n\Judgement: FALSE.

Statement: Are the legs a part of the car?\n\Judgement: FALSE.

Statement: Are the seeds a part of the zucchini?\n\Judgement: TRUE.

Statement: Is the sword a part of the blade?\n\Judgement: FALSE.

Statement: """,



"knowledge_5s" : """Knowledge: The wing is a part of the aeroplane.

Question: Is the aeroplane a part of the wing?\n\nAnswer: no.

Knowledge: The blade is a part of the sword.

Question: Is the sword a part of the blade?\n\nAnswer: no.

Knowledge: the gold is a part of the armchair.

Question: Is the gold a part of the armchair?\n\nAnswer: yes.

Knowledge: the fungus is a part of the television.

Question: Is the television a part of the fungus?\n\nAnswer: no.

Knowledge: The leg is a part of the body.

Question: Is the body a part of the leg?\n\nAnswer: no.

Knowledge: """


}