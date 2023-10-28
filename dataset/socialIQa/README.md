# SocialIQa dataset v1.4 (with dimensions)
Lines in each file are json encoded QA instances:
```json
{
 "context":"There was supposed to be a surprise quiz but Austin saw a copy of the quiz in the professor's briefcase so he knew about it already and passed easily.",
 "question":"How would Austin feel afterwards?",
 "answerA":"cheating by seeing the quiz in advance",
 "answerB":"unprepared for the quiz",
 "answerC":"like he cheated the system",
 "label_ix":2,
 "label_letter":"C",
 "promptDim":"xReact",
 "charmap": {"Austin":"X"},
 "answerSourcesOrigins":["qsa", "hia", "hca"],
 "answerSourcesWithCor":["qsa", "hia", "cor"],
 "promptQuestionFocusChar":"x"
}
```

where:
 -`promptDim`: ATOMIC dimension that was turned into a templated question, that was given to MTurk workers to embellish (the final question may not be related to the `promptDim`, but it gives an idea)
 -`promptQuestionFocusChar`: is the focus of the question related to the protagonist of the event (`x`) or other characters (`o`). See `charmap` for understanding who is who.
 -`charmap`: map of names to characters, used when filling in ATOMIC people placeholder (PersonX, PersonY, PersonZ). If `promptQuestionFocusChar` is `x` then the character mapping to `X` is the protagonist of the question, if it's `o` then the question is about others (`Y`, `Z`, or implied other participants)
 -`answerSourcesOrigins`: type of answer:
    - `hca`: handwritten correct answer
    - `hia`: handwritten incorrect answer
    - `qsa`: question-switched answer
 -`answerSourcesWithCor`: same as above, except the correct answer (i.e. at index `label_ix`)  is marked with `cor` 
