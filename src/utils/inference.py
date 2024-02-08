import tensorflow as tf
from .data_processing import portuguese_vectorizer, english_vectorizer
import numpy as np
from collections import Counter

word_to_id = tf.keras.layers.StringLookup(
    vocabulary=portuguese_vectorizer.get_vocabulary(),
    mask_token="",
    oov_token="[UNK]"
)

id_to_word = tf.keras.layers.StringLookup(
    vocabulary=portuguese_vectorizer.get_vocabulary(),
    mask_token="",
    oov_token="[UNK]",
    invert=True
)

unk_id = word_to_id("[UNK]")
sos_id = word_to_id("[SOS]")
eos_id = word_to_id("[EOS]")

print(unk_id, sos_id, eos_id)

def generate_next_token(decoder, context, next_token, done, state, temperature=0.0):
    # Get the logits and state from the decoder
    logits, state = decoder(context, next_token, state=state, return_state=True)

    logits = logits[:, -1, :]

    if temperature == 0.0:
        next_token = tf.argmax(logits, axis=-1)
    else:
        logits = logits / temperature
        next_token = tf.random.categorical(logits, num_samples=1)

    logits = tf.squeeze(logits)
    next_token = tf.squeeze(next_token)

    logit = logits[next_token].numpy()

    next_token = tf.reshape(next_token, shape = (1, 1))

    if next_token == eos_id:
        done = True

    return next_token, logit, state, done

def tokens_to_text(tokens, id_to_word):
    words = id_to_word(tokens)
    result = tf.strings.reduce_join(words, axis = -1, separator=" ")
    return result

def translate(model, text, UNITS=256, max_length=50, temperature=0.0):
    tokens, logits = [], []

    text = tf.convert_to_tensor(text)[tf.newaxis]
    context = english_vectorizer(text).to_tensor()
    context = model.encoder(context)

    next_token = tf.fill((1, 1), sos_id)
    state = [tf.zeros((1, UNITS)), tf.zeros((1, UNITS))]

    done = False

    for i in range(max_length):
        next_token, logit, state, done = generate_next_token(
            decoder=model.decoder,
            context=context,
            next_token=next_token,
            done=done,
            state=state,
            temperature=temperature
        )

        if done:
            break

        tokens.append(next_token)
        logits.append(logit)

    tokens = tf.concat(tokens, axis = -1)

    translation = tf.squeeze(tokens_to_text(tokens, id_to_word))
    translation = translation.numpy().decode()

    return translation, logits[-1], tokens

# MBR DECODING
# take several samples (random)
# score each sample against all other samples
# select the one with the highest score

def generate_samples(model, text, n_samples=4, temperature=0.6):
    samples, log_probs = [], []

    for _ in range(n_samples):
        _, logp, sample = translate(model, text, temperature=temperature)
        samples.append(np.squeeze(sample.numpy()).tolist())
        log_probs.append(logp)

    return samples, log_probs

def jaccard_similarity(candidate, reference):
    candidate_set = set(candidate)
    reference_set = set(reference)

    common_tokens = candidate_set.intersection(reference_set)
    all_tokens = candidate_set.union(reference_set)

    overlap = len(common_tokens) / len(all_tokens)

    return overlap

def rouge1_similarity(candidate, reference):
    candidate_word_counts = Counter(candidate)
    reference_word_counts = Counter(reference)

    overlap = 0

    for token in candidate_word_counts.keys():
        token_count_candidate = candidate_word_counts[token]
        token_count_reference = reference_word_counts[token]

        overlap += min(token_count_candidate, token_count_reference)

    precision = overlap / len(candidate)
    recall = overlap / len(reference)

    if precision + recall != 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score

    return 0

def average_overlap(samples, similarity_fn):
    scores = {}

    for index_candidate, candidate in enumerate(samples):
        overlap = 0
        for index_sample, sample in enumerate(samples):
            if (index_candidate == index_sample):
                continue

            sample_overlap = similarity_fn(candidate, sample)
            overlap += sample_overlap

        score = overlap / (len(samples) - 1)

        scores[index_candidate] = score

    return scores

l1 = [1, 2, 3]
l2 = [1, 2, 4]
l3 = [1, 2, 4, 5]
r1s = average_overlap([l1, l2, l3], jaccard_similarity)
print(r1s)

def mbr_decode(model, text, n_samples = 5, temperature=0.6, similarity_fn=rouge1_similarity):
    samples, log_probs = generate_samples(model, text, n_samples=n_samples, temperature=temperature)

    # compute the overlap scores
    scores = average_overlap(samples, similarity_fn)

    # decode samples
    decoded_translations = [tokens_to_text(s, id_to_word).numpy().decode("utf-8") for s in samples]

    # find the key with the highest score
    max_score_key = max(scores, key = lambda k: scores[k])

    translation = decoded_translations[max_score_key]

    return translation, decoded_translations