# Semantic cache: how we decide two questions are the same

Cells show **reused / wrong answers / extra Groq calls**. *Reused* = a true match answered from cache (good). *Wrong* = the user would be shown another question's answer (very bad). *Extra* = a true match we missed, which only costs one more Groq call (cheap).

Pair sets: **train** = `eval/question_pairs.csv` (47 pairs, 17 true matches); **holdout** = `eval/holdout_pairs.csv` (25 pairs, 12 true matches); **holdout2** = `eval/holdout2_pairs.csv` (38 pairs, 16 true matches). More than half of the pairs are deliberately tricky danger pairs, so wrong-answer counts here are much higher than real traffic should produce.

## The story, in order

1. **Score only.** The embedding model scores how close two questions are in meaning. But it treats a one-word change (Python vs JavaScript, open vs close, 2014 vs 2018) as almost the same question, so only a very high cutoff is safe, and that throws away most true matches. On the first set, 0.96 was the lowest cutoff with no wrong answers, reusing only 5 of 17 true matches.
2. **Word-list guard: overfit, caught by the holdout.** A list of names, numbers and opposite words fixed every danger pair it was built from (the train set became safe from 0.81 upward). But the holdout sets, written before the guard, had pairs it had no words for (zoom in/out 0.97, speed up/slow down, mute/unmute), and it still gave wrong answers. A word list only knows its own words, so only the numbers/years part (which did generalise) was kept.
3. **Cross-encoder verifier: did not clearly win, and costs RAM.** A small model (`cross-encoder/quora-distilroberta-base`, trained on Quora duplicate questions) reads both questions together. It rejected many danger pairs the others accepted, but it still called some opposites the same question (mute/unmute 0.99), rejected many real paraphrases, never reached zero wrong answers, and adds 105 MB of RAM (365 MB total vs 260 MB). It is used only for evaluation, never loaded by the server.
4. **Shipped: cutoff 0.96 + numbers guard, with mitigations for the wrong answers that remain.** (a) The `/chat` response now includes `matched_question`, the saved question whose answer was reused, so a caller or user can see what was matched. (b) Callers can send `"semantic_cache": false` to skip the semantic step (the exact cache still applies). (c) Every call, including near-misses, logs its `similarity`, so real traffic can be audited.

## Final table (train / holdout / holdout2)

| approach | train | holdout | holdout2 |
|---|---|---|---|
| (a) score only: embedding cutoff 0.85 | 12/17 · 6 · 5 | 10/12 · 6 · 2 | 16/16 · 17 · 0 |
| (a) score only: embedding cutoff 0.90 | 11/17 · 5 · 6 | 7/12 · 4 · 5 | 14/16 · 9 · 2 |
| (a) score only: embedding cutoff 0.96 | 5/17 · 0 · 12 | 3/12 · 1 · 9 | 7/16 · 5 · 9 |
| (a) score only: embedding cutoff 0.99 | 0/17 · 0 · 17 | 0/12 · 0 · 12 | 2/16 · 3 · 14 |
| (b) + word-list guard, cutoff 0.96 | 5/17 · 0 · 12 | 3/12 · 1 · 9 | 6/16 · 5 · 10 |
| (c) + verifier (embedding ≥ 0.85) + numbers guard, threshold 0.97 | 5/17 · 0 · 12 | 4/12 · 2 · 8 | 8/16 · 4 · 8 |
| **(e) SHIPPED: embedding cutoff 0.96 + numbers guard** | 5/17 · 0 · 12 | 3/12 · 1 · 9 | 6/16 · 5 · 10 |

## Decision

**No approach reached zero wrong answers on all three sets** (score only, word-list guard, verifier, or the shipped setting). The verifier did not clearly beat the simpler options, so we ship embedding cutoff 0.96 + numbers guard, plus the mitigations above. A cutoff of 0.99 is safer but reuses almost nothing.

## Verifier cost

`cross-encoder/quora-distilroberta-base` as an 8-bit ONNX file: **83 MB** on disk (+3.6 MB tokenizer). RAM for the whole app process: **93 MB** with no models, **260 MB** with the bge-small embedder, **365 MB** with the verifier added (+105 MB), which leaves little room on Render's 512 MB free tier. Speed: **about 13 ms per pair** (max 29 ms), only for candidates above the prefilter, plus about 1.8 s to load at startup.

## Honest caveats

- **Nothing reached zero wrong answers on all three sets**, with or without the guard or the verifier. The hardest pairs swap two things around (kilometers to miles vs miles to kilometers, 5 dollars in rupees vs 5 rupees in dollars) or change singular to plural; the embedding scores them 0.99+ and even the verifier is fooled by some.
- **The verifier helps but is not enough.** It rejects many danger pairs the other approaches accept (Celsius/Fahrenheit, km/miles, enable/disable), but it calls some opposites the same question (mute/unmute 0.99, zoom in/out 0.98, pros/cons 0.97) and rejects many real paraphrases ("What is RAG?" vs "Explain RAG to me" scores 0.002). It also costs 105 MB of RAM.
- **These sets are deliberately adversarial**: more than half the pairs are danger pairs built to fool the cache. Real users ask far fewer one-word-different questions, so real-traffic wrong-answer rates should be much lower than these counts. Check the `similarity` column in the `calls` table on live traffic.
- **One label was corrected after the first holdout2 run**: "before eating or after eating" vs "after eating or before eating" is the same question in a different word order, so it was changed from *no* to *yes* (my labelling mistake). Every other label is unchanged.
- **Small sets**: 47, 25 and 38 pairs. A difference of one or two pairs is noise.

## Pairs that must NOT match, with their scores (embedding ≥ 0.85)

| set | embedding | verifier prob | numbers guard | word-list guard | pair |
|---|---|---|---|---|---|
| train | 0.954 | 0.000 | — | different named terms | How do I reverse a list in Python? ↔ How do I reverse a list in JavaScript? |
| train | 0.939 | 0.001 | — | different named terms | How do I install Docker on Ubuntu? ↔ How do I install Docker on Windows? |
| train | 0.905 | 0.003 | — | opposite actions | How do I open a port in the firewall? ↔ How do I close a port in the firewall? |
| train | 0.904 | 0.968 | — | opposite actions | How do I encrypt a file? ↔ How do I decrypt a file? |
| train | 0.903 | 0.004 | different numbers | different numbers | Give me 3 tips for studying ↔ Give me 10 tips for studying |
| train | 0.852 | 0.000 | different numbers | different numbers | Who won the FIFA World Cup in 2014? ↔ Who won the FIFA World Cup in 2018? |
| holdout | 0.972 | 0.981 | — | **missed** | How do I zoom in on the map? ↔ How do I zoom out on the map? |
| holdout | 0.928 | 0.004 | — | **missed** | How do I speed up my laptop? ↔ How do I slow down my laptop? |
| holdout | 0.910 | 0.003 | — | **missed** | how do I sort a list in haskell? ↔ how do I sort a list in scala? |
| holdout | 0.904 | 0.991 | — | **missed** | How do I mute the microphone? ↔ How do I unmute the microphone? |
| holdout | 0.898 | 0.000 | different numbers | different numbers | How do I update Windows 10? ↔ How do I update Windows 11? |
| holdout | 0.860 | 0.001 | different numbers | different numbers | Who won the Nobel Prize in Physics in 2019? ↔ Who won the Nobel Prize in Physics in 2021? |
| holdout2 | 0.997 | 0.024 | — | **missed** | How do I convert Celsius to Fahrenheit? ↔ How do I convert Fahrenheit to Celsius? |
| holdout2 | 0.996 | 0.015 | — | **missed** | How do I convert kilometers to miles? ↔ How do I convert miles to kilometers? |
| holdout2 | 0.991 | 0.992 | — | **missed** | How much is 5 dollars in rupees? ↔ How much is 5 rupees in dollars? |
| holdout2 | 0.987 | 0.993 | — | **missed** | What is a good way to learn a language? ↔ What are good ways to learn a language? |
| holdout2 | 0.981 | 0.992 | — | **missed** | How do I write a unit test? ↔ How do I write unit tests? |
| holdout2 | 0.938 | 0.011 | — | opposite actions | How do I enable two-factor authentication? ↔ How do I disable two-factor authentication? |
| holdout2 | 0.937 | 0.042 | — | **missed** | How many grams are in an ounce? ↔ How many ounces are in a gram? |
| holdout2 | 0.936 | 0.974 | — | **missed** | What are the pros of remote work? ↔ What are the cons of remote work? |
| holdout2 | 0.925 | 0.164 | — | **missed** | Is coffee good for you? ↔ Is coffee bad for you? |
| holdout2 | 0.889 | 0.006 | — | **missed** | How long is a marathon in kilometers? ↔ How long is a marathon in miles? |
| holdout2 | 0.887 | 0.000 | — | different named terms | Translate hello into Spanish ↔ Translate hello into French |
| holdout2 | 0.883 | 0.056 | — | **missed** | How do I find the minimum value in a list? ↔ How do I find the maximum value in a list? |
| holdout2 | 0.873 | 0.075 | — | opposite actions | How do I turn on Bluetooth on my phone? ↔ How do I turn off Bluetooth on my phone? |
| holdout2 | 0.872 | 0.001 | — | opposite actions | How can I increase my website traffic? ↔ How can I decrease my website traffic? |
| holdout2 | 0.865 | 0.007 | — | **missed** | How do I fix a bug in my code? ↔ How do I find all the bugs in my code? |
| holdout2 | 0.860 | 0.002 | — | **missed** | What is the speed of light in meters per second? ↔ What is the speed of light in miles per hour? |
| holdout2 | 0.852 | 0.002 | — | **missed** | How do I get a refund for my order? ↔ How do I cancel my order? |

## True matches not reused by the verifier setting (threshold 0.9); cost: an extra Groq call

| set | embedding | verifier prob | pair |
|---|---|---|---|
| train | 0.501 | (not run: below prefilter) | What is RAG? ↔ What is retrieval augmented generation? |
| train | 0.597 | (not run: below prefilter) | What is CI/CD? ↔ What is continuous integration and continuous delivery? |
| train | 0.640 | (not run: below prefilter) | Explain LLMs ↔ Explain large language models |
| train | 0.721 | (not run: below prefilter) | What does SQL stand for? ↔ What does Structured Query Language mean? |
| train | 0.773 | (not run: below prefilter) | What is NLP? ↔ What is natural language processing? |
| train | 0.891 | 0.001 | How do I undo my last git commit? ↔ How can I revert the most recent commit in git? |
| train | 0.909 | 0.341 | How do I make a POST request with Python requests? ↔ How to send a POST request using the Python requests library? |
| train | 0.911 | 0.002 | What is RAG? ↔ Explain RAG to me |
| holdout | 0.707 | (not run: below prefilter) | What does AWS stand for? ↔ What is the full form of Amazon Web Services? |
| holdout | 0.776 | (not run: below prefilter) | What is a GPU? ↔ What is a graphics processing unit? |
| holdout | 0.866 | 0.000 | What is the time complexity of binary search? ↔ How fast is binary search in big O terms? |
| holdout | 0.876 | 0.079 | Who is the president of the United States? ↔ Who currently serves as president of the United States? |
| holdout | 0.883 | 0.000 | What is the boiling point of water? ↔ At what temperature does water boil? |
| holdout | 0.935 | 0.888 | How do I convert a string to an integer in Python? ↔ What's the way to turn a string into an int in Python? |
| holdout | 0.939 | 0.042 | How do I check my Python version? ↔ How can I find out which Python version I have? |
| holdout2 | 0.897 | 0.020 | How do I add a column in Excel? ↔ How can I insert a new column in an Excel sheet? |
| holdout2 | 0.918 | 0.802 | What causes inflation? ↔ Why does inflation happen? |
| holdout2 | 0.927 | 0.293 | What is the boiling point of water in Fahrenheit? ↔ At what Fahrenheit temperature does water boil? |
| holdout2 | 0.936 | 0.039 | How do I reset my router? ↔ How can I restart my router? |
| holdout2 | 0.970 | 0.088 | What is the meaning of life? ↔ What is life's meaning? |
| holdout2 | 0.972 | 0.984 | How do I write a good cover letter? ↔ What makes a good cover letter and how do I write one? |
