# src/client/query_processor.py
import ast
import re
from spellchecker import SpellChecker
import ollama
from src.config import LLM_MODEL

# Load spell checker and add custom wizard words
spell = SpellChecker()
custom_words = [
    'harry', 'potter', 'hermione', 'ron', 'weasley', 'dumbledore', 'voldemort',
    'snape', 'malfoy', 'gryffindor', 'slytherin', 'hufflepuff', 'ravenclaw',
    'quidditch', 'patronus', 'horcrux', 'muggle', 'mudblood', 'death', 'eater',
    'hogwarts', 'diagon', 'azkaban', 'pensieve', 'accio', 'expecto', 'lumos',
    'wingardium', 'leviosa', 'riddikulus', 'alohomora', 'expelliarmus',
    'crucio', 'imperio', 'avada', 'kedavra', 'sectumsempra', 'legilimency',
    'occlumency', 'animagus', 'werewolf', 'hippogriff', 'thestral', 'basilisk',
    'grawp', 'kreacher', 'dobby', 'winky', 'luna', 'neville', 'ginny', 'fred',
    'george', 'percy', 'bill', 'charlie', 'molly', 'arthur', 'lupin', 'tonks',
    'kingsley', 'moody', 'sirius', 'james', 'lily', 'petunia', 'vernon', 'dudley',
    'severus', 'patronus', 'patranous',  # include common misspellings
]
for word in custom_words:
    spell.word_frequency.add(word)

def correct_spelling(text):
    corrected = []
    for w in text.split():
        # Leave capitalised words alone: they are usually names that the
        # dictionary does not know and would otherwise be "corrected" away.
        if w.isalpha() and not w[0].isupper() and w.lower() not in spell:
            correction = spell.correction(w)
            corrected.append(correction or w)
        else:
            corrected.append(w)
    return ' '.join(corrected)


def _parse_string_list(text):
    """Safely parse an LLM-produced Python/JSON list of strings.
    (Previously used eval(), which executes arbitrary model output.)"""
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if not match:
        return []
    try:
        value = ast.literal_eval(match.group(0))
    except (ValueError, SyntaxError):
        return []
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if isinstance(v, (str, int, float)) and str(v).strip()]

def rewrite_query_with_llm(query):
    prompt = f"""You are a query rewriter. The user asked: "{query}"
Rewrite this query into a clear, complete, and explicit question that a document retrieval system can use to find relevant information. If the query is already clear, return it unchanged.
Only output the rewritten query, nothing else."""
    try:
        response = ollama.chat(model=LLM_MODEL, messages=[{'role': 'user', 'content': prompt}])
        rewritten = response['message']['content'].strip().strip('"')
        return rewritten or query
    except Exception as e:
        print(f"[REWRITE] Failed, using original query: {e}")
        return query

def expand_query(query):
    prompt = f"""Generate 3 different rephrasings of the following question, each focusing on a different aspect or using different wording. Keep the meaning but vary the phrasing.
Question: {query}
Output as a Python list of strings, e.g. ["...", "...", "..."].
Only output the list, no extra text."""
    try:
        response = ollama.chat(model=LLM_MODEL, messages=[{'role': 'user', 'content': prompt}])
        return _parse_string_list(response['message']['content']) or [query]
    except Exception:
        return [query]

def generate_keywords(query):
    """Generate alternative search terms and related concepts."""
    prompt = f"""Given the user's question: "{query}"
Generate a list of alternative search terms, phrases, and related concepts that could appear in documents relevant to answering this question.
Output as a Python list of strings, e.g. ["term1", "phrase 2", "concept"].
Only output the list, no extra text."""
    try:
        response = ollama.chat(model=LLM_MODEL, messages=[{'role': 'user', 'content': prompt}],
                               options={"num_predict": 128})
        return [k.lower() for k in _parse_string_list(response['message']['content'])][:15]
    except Exception:
        return []

def preprocess_query(query, use_llm_rewrite=True):
    corrected = correct_spelling(query.strip())
    return rewrite_query_with_llm(corrected) if use_llm_rewrite else corrected