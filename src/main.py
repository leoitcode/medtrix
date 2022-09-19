import streamlit as st
import warnings
warnings.filterwarnings("ignore")
from detector import get_ents_input_text, get_ents_input_text_vis
from similarity import get_similar_document
from replace import fake_phi_labels, pretty_print_mimic
from spacy_streamlit import visualize_spans

if __name__ == "__main__":
    st.title("MedtriX")
    st.header("Generating Medical Notes from free text")
    text = st.text_input("Describe a patient report:")
    if text:
        st.subheader("Detected Entities:")

        input_text_ents  = get_ents_input_text(text)
        doc, options = get_ents_input_text_vis(input_text_ents, text)
        visualize_spans(doc, displacy_options=options, show_table=False, title=False)

        if input_text_ents:
            st.subheader("Medical Document:")
            sections_text = get_similar_document(input_text_ents)
            sections_text = fake_phi_labels(sections_text, **input_text_ents)
            final_text = pretty_print_mimic(sections_text)
            st.markdown(final_text)
            