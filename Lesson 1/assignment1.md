# Assignment 1 — Your First LLM App

**Lecture 1 · Intro to AI & LLMs**

By the end of these two exercises you will have (1) your own Python code calling a hosted LLM and (2) a small open-source model running locally on your machine. This is step 1 of the app you'll grow over the whole course.

You'll write the code yourself — the tasks below tell you *what* to build and *which* tools to use, not the exact lines.

> **Pick your own theme.** This is the start of a project you'll grow all course, so choose a domain that interests you (recipes, sports, personal finance, a game, your own docs — anything). Use it to shape your prompts, your example questions, and the file you use in Exercise 1c.

> **Start a fresh repo.** Create a **new Git repository** for this part of the course (separate from anything earlier), commit your work there, and **push it to a remote** (e.g. GitHub). You'll keep building on this same repo in the coming lectures.

---

## Setup (do this once)

**Recommended:** use **Python 3.12 or newer**, and work inside a **virtual environment** so this assignment's packages stay isolated from the rest of your system:

```bash
python3 --version                # confirm 3.12+
python3 -m venv .venv            # create the environment
source .venv/bin/activate        # macOS / Linux  (prompt should show (.venv))
# .venv\Scripts\activate         # Windows
```

Activate it again in any new terminal before running your scripts.

Then install the packages you'll need:

```bash
pip install openai pydantic
pip install "transformers>=4.44" torch
```

You'll need a **Claude (Anthropic) API key** for Exercise 1. Set it as an environment variable so it never lives in your code:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."       # macOS / Linux
# setx ANTHROPIC_API_KEY "sk-ant-..."        # Windows (then open a new terminal)
```

> We use the **OpenAI library**, but point it at **Anthropic's OpenAI-compatible endpoint** — so the exact same client code talks to Claude instead of GPT. The API shape is near-universal across providers; swapping providers is roughly a change of `base_url` + `api_key` + `model`.

---

## Exercise 1 — Hello, LLM (Claude, via the OpenAI library)

**Goal:** call a hosted model from Python and print its reply.

**Build a script `hello_llm.py` that:**

- **Library / package:** uses the `openai` package.
- **Client:** creates an OpenAI client, but configured to talk to Claude:
  - set **`base_url`** to Anthropic's OpenAI-compatible endpoint: `https://api.anthropic.com/v1/`
  - set **`api_key`** to your Claude key — read it from the environment (`os.environ["ANTHROPIC_API_KEY"]`), don't hard-code it
- **The call:** sends a chat-completion request with two things:
  - a **model** — use a small, cheap Claude model like `claude-haiku-4-5`
  - a **messages** list — a single user message, e.g. `"Say hello in one sentence."`
- **Output:** extracts the text of the model's reply from the response object and prints it.

Run it:

```bash
python hello_llm.py
```

✅ **Done when:** you see a one-sentence reply printed in your terminal. That's your first LLM app.

> Hint: everything you need is on the "Anatomy of an API call" slide — the four levers are `model`, `messages`, `temperature`, `max_tokens`. You only need the first two to start. The only Claude-specific parts are `base_url` and `api_key` on the client, plus the Claude `model` name.

---

## Exercise 1b — Now *engineer* it

Extend `hello_llm.py`. Change **one thing at a time** and observe the effect — this is prompt & context engineering in miniature.

1. **Add a system prompt.** Put a second message with the `system` role *before* the user message, giving the model a persona and rules (e.g. "You are a pirate. Answer in one sentence."). → Notice how the *same* user message now produces a different style.

2. **Play with temperature.** Add a `temperature` parameter to the call. Run with `temperature=0` twice, then `temperature=1` twice. → `0` should be (nearly) identical each run; `1` should vary. That's determinism vs. creativity.

3. **Ask for structured output — two ways.** Change the system prompt to instruct the model to reply with **JSON only** — e.g. an object with an `answer` (string) and a `confidence` (number 0–1). Then get that JSON into Python two different ways and compare:
   - **(a) Raw JSON:** parse the reply string with Python's built-in `json` module and print the individual fields.
   - **(b) Pydantic:** define a `pydantic` model — a `BaseModel` subclass with typed fields (`answer: str`, `confidence: float`) — and load the reply into it (e.g. `Answer.model_validate_json(...)`). Now you get **validation for free**: if the model returns the wrong shape or a bad type, Pydantic raises an error instead of silently passing bad data downstream.

   → A typed, validated object, not prose — this is what turns a chatbot into a software *component*.

✅ **Done when:** you've seen the system prompt steer the output, seen temperature `0` vs `1` differ, and parsed a JSON reply into Python values both with `json` and with a Pydantic model.

**Reflect:** which change had the biggest effect on the output? Why?

---

## Exercise 1c — Answer questions about a file (grounded)

**Goal:** turn your LLM call into a tiny question-answering tool that answers *only* from a file you give it — and admits when it doesn't know instead of guessing. (This is a first taste of RAG, coming in Lecture 3.)

**Build a script `file_qa.py` that:**

- **Loads a file:** reads a text file from disk (take the path from a command-line argument, e.g. `sys.argv`, or `input()`). Read its contents into a string.
- **Takes a question:** gets a question from the user (another command-line argument or `input()`).
- **Builds the context:** sends the model a **system prompt** that sets the rules, plus a **user message** containing both the file contents and the question. The system prompt must tell the model to:
  - answer **only** using the provided document — do **not** use outside knowledge or guess
  - **quote the exact passage** from the document that supports the answer (the reference)
  - if the answer is **not** in the document, reply with a fixed sentence like *"I can't find that in the document."* and nothing else
- **Prints** the model's answer.
- Reuse the same Claude client setup from Exercise 1 (`base_url`, `api_key`, a `claude-*` model).

**Recommendations:**

1. **Use a small file (~1000 words).** The whole file goes into the prompt, and you pay per token — a short file keeps the cost tiny while you experiment.
2. **Use a plain-text format.** Work with a `.txt` or `.md` file. If your source is a PDF, Word doc, etc., convert it to text first using any free online converter.

Run it, for example:

```bash
python file_qa.py notes.txt "What is the main topic?"
python file_qa.py insurance.md "Does my insurance cover glasses for kids?"
```

Try three kinds of questions and watch the behavior:

1. A question clearly answered in the file → you should get an answer **plus a quoted reference**.
2. A question the file doesn't cover → the model should say it can't find it, **not** invent an answer.
3. A trick question that sounds related but isn't in the file → it should still refuse to guess.

✅ **Done when:** your tool answers grounded questions with a quote, and reliably says *"I can't find that in the document."* for anything the file doesn't cover.

**Reflect:** why does forcing a quote and allowing "I don't know" reduce hallucination?

---

## Exercise 2 — Run an open model locally (Hugging Face)

**Goal:** run a small open-source model on your own machine — no API key, no per-token bill, nothing leaves your computer.

**Build a script `local_llm.py` that:**

- **Library / packages:** uses `transformers` (which uses `torch` under the hood — both already installed).
- **Load the model:** create a Hugging Face **`pipeline`** for the `"text-generation"` task, pointing it at a small instruct model — use `Qwen/Qwen2.5-0.5B-Instruct` (tiny on purpose, so it runs on a laptop CPU).
- **Generate:** pass a prompt (e.g. `"Give me a recepie for pizza without gluten."`) to the pipeline.
- **Output:** pull the generated text out of the pipeline's result and print it.

Run it:

```bash
python local_llm.py
```

> The **first run downloads the model (~1 GB)** — needs internet once, then runs fully offline. It will be slower and weaker than the hosted model — that's the point.

✅ **Done when:** the model loads and prints a reply generated entirely on your machine.

---

## Reflect — closed vs. open

You just ran both. In a sentence or two each:

- **Closed (OpenAI):** what was easy? what did it cost you (money, data leaving your machine, rate limits)?
- **Open (local HF):** what did you gain (privacy, no per-call cost, control)? what did you pay in (speed, quality, setup, RAM)?

There's no universal winner — the choice is a **tradeoff**, and now you've lived it.

---

## What to submit / show

- `hello_llm.py` (with your 1b modifications)
- `file_qa.py` (Exercise 1c)
- `local_llm.py`
- `reflections.md` — a markdown file answering **all** the reflection prompts from the exercises:
  - **1b:** which change (system prompt / temperature / structured output) had the biggest effect on the output, and why?
  - **1c:** why does forcing a quote and allowing "I don't know" reduce hallucination?
  - **Closed vs. open:** the two sentences from the "Reflect — closed vs. open" section above.

**Commit everything to your new course repo and push it to your remote** — that repo is where the rest of the course's work will live.
