from pathlib import Path
import html
import json
import re
import pandas as pd
import streamlit as st
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
st.set_page_config(page_title='Catalyst Consulting | Executive Overview', page_icon='📊', layout='wide', initial_sidebar_state='collapsed')

st.markdown('''<style>
.block-container {padding-top: 1.6rem; max-width: 1320px}
h1 {font-size: 2.35rem !important}
[data-testid="stMetricValue"] {font-size: 2.1rem}
.issue-wrap {display:flex;flex-wrap:wrap;gap:18px;align-items:center;justify-content:center;padding:16px 0}
.issue {display:flex; flex-direction:column; justify-content:center; align-items:center; text-align:center;
 border-radius:50%; padding:20px; flex-shrink:0; box-sizing:border-box; line-height:1.25;}
.issue-title {font-size:1.05rem;font-weight:700;max-width:90%;}
.issue-count {font-size:0.9rem;margin-top:7px}
</style>''', unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv(RESULTS / 'error_analysis_full.csv')
    df['text'] = df['text'].fillna('').astype(str)
    df['rating_sentiment'] = df['rating_sentiment'].fillna('').astype(str).str.lower()
    with open(RESULTS / 'vader_full_metrics.json', encoding='utf-8') as f: vader = json.load(f)
    with open(RESULTS / 'transformer_full_metrics.json', encoding='utf-8') as f: roberta = json.load(f)
    return df, vader, roberta

@st.cache_data(show_spinner='Finding the recurring themes in customer complaints…')
def model_issues(review_ids, review_texts, n_topics):
    # Topics are unsupervised clusters; labels are keywords, not verified issue types.
    cleaned = [re.sub(r'<[^>]+>', ' ', html.unescape(str(t))) for t in review_texts]
    vectorizer = TfidfVectorizer(stop_words='english', max_features=6000, min_df=3,
                                  max_df=0.9, ngram_range=(1, 2), strip_accents='unicode')
    X = vectorizer.fit_transform(cleaned)
    count = min(n_topics, X.shape[0], X.shape[1])
    if count < 2: raise ValueError('Not enough review text for topic modeling.')
    model = NMF(n_components=count, init='nndsvda', random_state=42, max_iter=400)
    assignments = model.fit_transform(X).argmax(axis=1)
    terms = vectorizer.get_feature_names_out()
    results = []
    for i, component in enumerate(model.components_):
        top = [terms[j] for j in component.argsort()[-3:][::-1]]
        results.append({'topic_id':i, 'theme': ' / '.join(top[:2]), 'keywords': ', '.join(top),
                        'reviews': int((assignments == i).sum())})
    topics = pd.DataFrame(results).sort_values('reviews', ascending=False).reset_index(drop=True)
    assigned = pd.DataFrame({'review_id':list(review_ids), 'topic_id':assignments})
    return topics, assigned

st.caption('CATALYST CONSULTING  /  CUSTOMER INTELLIGENCE')
st.title('What customers are struggling with')
st.write('A decision-ready view of recurring themes in customer complaints.')

try:
    df, vader, roberta = load_data()
except (FileNotFoundError, ValueError, KeyError) as exc:
    st.error(f'Missing evaluation files: {exc}. Run the existing analysis scripts first.')
    st.stop()

negative = df.loc[df['rating_sentiment'].eq('negative'), ['review_id','text']].dropna().copy()
if negative.empty:
    st.warning('There are no negative-rated reviews to analyze.')
    st.stop()

with st.sidebar:
    st.header('Overview settings')
    number_topics = st.slider('Number of discovered themes', 4, 10, 6)
    st.caption('Themes are discovered automatically and should be reviewed by an analyst before making operational decisions.')

try:
    topics, topic_assignments = model_issues(tuple(negative['review_id'].astype(str)), tuple(negative['text']), number_topics)
except ValueError as exc:
    st.error(f'Unable to model complaint themes: {exc}')
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric('Customer reviews analyzed', f'{len(df):,}')
c2.metric('Negative-rated reviews', f'{len(negative):,}')
c3.metric('Recurring complaint themes', f'{len(topics)}')

st.markdown('### Biggest discussion areas')
st.caption('Each review is assigned to one dominant topic. Bigger, darker-red ovals = more negative-rated reviews in that topic; this is frequency, not severity.')

max_count = max(int(topics['reviews'].max()), 1)
min_count = int(topics['reviews'].min())
parts = ['<div class="issue-wrap">']
for row in topics.itertuples(index=False):
    rel = (row.reviews - min_count) / max(max_count-min_count, 1)
    width = int(180 + 170 * rel)
    height = int(112 + 86 * rel)
    bg = f'rgb({int(240-60*rel)},{int(166-125*rel)},{int(166-125*rel)})'
    fg = '#ffffff' if rel >= 0.46 else '#521717'
    label = html.escape(row.theme)
    parts.append(f'<div class="issue" style="width:{width}px;height:{height}px;background:{bg};color:{fg}">'
                 f'<div class="issue-title">{label}</div><div class="issue-count">{row.reviews:,} reviews</div></div>')
parts.append('</div>')
st.markdown(''.join(parts), unsafe_allow_html=True)

st.markdown('### Where to investigate first')
for rank, row in enumerate(topics.head(3).itertuples(index=False), 1):
    share = 100 * row.reviews / len(negative)
    st.markdown(f'**{rank}. {row.theme}** — {row.reviews:,} reviews ({share:.1f}% of negative-rated reviews)')

st.caption('These are algorithmically discovered keyword-based clusters, not validated product defect categories. Review examples before assigning ownership or prioritizing fixes.')

st.markdown('### Explore a discussion area')
chosen = st.selectbox('Choose a theme', topics['topic_id'].tolist(),
                      format_func=lambda i: topics.loc[topics.topic_id.eq(i), 'theme'].iloc[0])
selected_ids = topic_assignments.loc[topic_assignments.topic_id.eq(chosen), 'review_id']
examples = negative[negative.review_id.astype(str).isin(selected_ids)].head(5)
for r in examples.itertuples(index=False):
    with st.expander(f'Review {r.review_id}'):
        st.write(r.text)

st.divider()
a, b = st.columns([3, 1])
with a:
    st.markdown('**Need the evidence behind these trends?**')
    st.caption('Classification metrics, confusion matrices, full review explorer, and diagnostic flags live in the technical view.')
with b:
    if st.button('Analysts & Engineers →', type='primary', use_container_width=True):
        st.switch_page('pages/1_Analysts_and_Engineers.py')

st.caption('Research prototype • Local customer-review dataset • No evidence of issue severity or real-world failure rates from this graphic alone')
