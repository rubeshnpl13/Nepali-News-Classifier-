import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from predict import (
    LABEL_NAMES,
    VOCAB_PATH,
    choose_device,
    load_model,
    CharacterTokenizer,
    predict_text,
)


st.set_page_config(
    page_title="Nepali News Classifier",
    page_icon="📰",
    layout="centered",
)


@st.cache_resource
def load_prediction_components():
    device = choose_device()
    tokenizer = CharacterTokenizer.load(VOCAB_PATH)
    model = load_model(device)

    return model, tokenizer, device


def display_label(label_name):
    display_names = {
        "general_society": "General / Society",
        "culture_entertainment": (
            "Culture / Entertainment"
        ),
        "sports": "Sports",
    }

    return display_names[label_name]


st.title("Nepali News Topic Classifier")

st.write(
    "Type or paste a Nepali news article below. "
    "The model will predict its topic."
)

st.caption(
    "Model: character-level Transformer built from scratch"
)

example_text = (
    "नेपालले नयाँ खेलकुद प्रतियोगिता आयोजना "
    "गर्ने भएको छ।"
)

news_text = st.text_area(
    "Nepali news text",
    value="",
    height=220,
    placeholder=example_text,
)

classify_clicked = st.button(
    "Classify news",
    type="primary",
)

if classify_clicked:
    if not news_text.strip():
        st.warning("Please enter some Nepali news text.")

    else:
        with st.spinner("Classifying..."):
            model, tokenizer, device = (
                load_prediction_components()
            )

            result = predict_text(
                text=news_text,
                model=model,
                tokenizer=tokenizer,
                device=device,
            )

        predicted_label = display_label(
            result["label_name"]
        )

        st.subheader("Prediction")
        st.success(predicted_label)

        st.subheader("Class probabilities")

        for label_name, probability in (
            result["probabilities"].items()
        ):
            st.write(display_label(label_name))
            st.progress(
                probability,
                text=f"{probability:.2%}",
            )

        st.subheader("Input details")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Character count",
                len(news_text),
            )

        with col2:
            st.metric(
                "Unknown characters",
                result["unknown_character_count"],
            )

        if result["token_count"] > 320:
            st.warning(
                "This text is longer than the model's "
                "maximum sequence length. It was truncated."
            )

        st.info(
            "This is a small educational model. "
            "Probabilities are model scores, not certainty."
        )

st.divider()

st.subheader("Supported categories")

for class_id, label_name in LABEL_NAMES.items():
    st.write(
        f"{class_id}: {display_label(label_name)}"
    )