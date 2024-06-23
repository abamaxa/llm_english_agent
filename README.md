# Demo LLM-based English Improvement Agent

## Setup using Docker

First build the docker image for the agent:

```shell
$ docker build . -t llm_english_agent
```

An OpenAI API key is needed to run the agent. If that is already present in your environment as `OPENAI_API_KEY`, you can run the agent with:

```shell
docker run -it --rm -e OPENAI_API_KEY llm_english_agent
```

If the API key is present in an environment file you pass the location of that file to docker:

```shell
docker run -it --rm --env-file <path-to-env-file> llm_english_agent
```

Otherwise, you can pass the API on the command line, however this is not recommended as your key will then be stored in your shell history.

```shell
docker run -it --rm -e OPENAI_API_KEY=<your key> llm_english_agent
```

Once the agent starts, follow the instructions on screen.

## Technical Questions

● Prompt Engineering for RAG: Describe how you would design prompts for the RAG model to differentiate between 'write_properly' and 'write_the_same_grammar_fixed' functions. What considerations will you make to ensure the model understands the distinction between style enhancement and mere grammatical correction?

I really needed clearer definitions of what the functions 'write_properly' and 'write_the_same_grammar_fixed' were supposed to do. To me, writing styles could be formal, informal, humorous, sarcastic, etc. I decided that it meant the system should output Standard English. However, it was very unclear to me where to draw the line between grammar and style. I tried to implement prompts that would only correct grammar, rather than both style and grammar, using techniques such as few-shot learning, but without much success.

I searched Hugging Face for datasets to use for a RAG (Retrieval-Augmented Generation) knowledge base but didn't have much success. Eventually, I put together a set of simple grammar rules from the internet and hoped the magic of vector searching would match relevant rules, but I don't think that worked very well.

● API Utilization Strategy: Explain your strategy for utilizing the OpenAI API in this task. How will you ensure efficient and effective use of the API for different functions, especially considering the potential variability in the length and complexity of user inputs?

I didn't have time to implement handling of large inputs. If I had, I would have implemented chunking based on paragraphs, or blocks of sentences if the paragraphs turned out to be too large to fit in the context window. My wrapper around the OpenAI API writes the responses received from the API to a directory so that they can be analysed later, using pandas for instance.

● Handling Ambiguity in User Inputs: Given that the users are non-native English speakers, their inputs may contain ambiguities or context-specific nuances. How will you design the system to handle such ambiguities, especially in the 'write_properly' function where style improvement is also considered?

I included instructions in the prompts telling the model to think step by step and, if the text was ambigous, to output the text "AMBIGOUS TEXT" followed by a clear and concise explanation of the problem and to then stop processing.

● Summarization Technique: Discuss the approach you will take to implement the 'Summarize' function. How will you ensure that the essential points are retained while maintaining brevity, and how does this approach leverage the RAG model?

I used a Huggingface transformer to implement the 'Summarize' function, using the t5-small model which I believe has been optimized to extract the essential points from the text.

● Performance Metrics and Evaluation: What metrics would you use to evaluate the performance of each function in this application? How would you gather feedback or data to improve the system continuously, and what role does prompt engineering play in this process?

I found datasets on Huggingface for training grammar correction models and tried to use them to assess the quality of the system, but I didn't have time to implement it.